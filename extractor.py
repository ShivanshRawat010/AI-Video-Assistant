from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import Runnable
from dotenv import load_dotenv

load_dotenv()

_llm = None

def load_model():
  global _llm

  if _llm is None:
    _llm = ChatMistralAI(temperature=0.2)

  return _llm

def build_chain(system_prompt : str):

  llm = load_model()
  parser = StrOutputParser()

  prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human" , "{text}")
  ])

  return prompt | llm | parser

def chunk_transcript(transcript : str) -> list:

  splitter = RecursiveCharacterTextSplitter(
    chunk_size = 2000,
    chunk_overlap = 200
  )

  return splitter.split_text(transcript)

def reduce_transcript(chain : Runnable, transcript : str, none_case : str) -> str:
  chunks = chunk_transcript(transcript)
  chunk_results = []

  for chunk in chunks:
    result = chain.invoke({"text" : chunk})

    if result.strip() != none_case:
      chunk_results.append(result)

  if not chunk_results:
    return none_case

  return "\n\n".join(chunk_results)


def extract_action_items(transcript : str) -> str:
  map_chain = build_chain(
    """
    You are an expert meeting analyst.

    Extract all action items from this portion of the meeting transcript.

    For each action item provide:
    - Task description
    - Owner (or 'Not specified')
    - Deadline (or 'Not specified')

    If none are found, return 'No action items found.'
    """
  )

  reduce_chain = build_chain(
    """
    You are an expert meeting analyst.

    You are given action items extracted from different portions of the same meeting.

    Merge duplicate action items.
    Combine partial information.
    Preserve owners and deadlines.
    Return a clean numbered list.

    If no action items exist, return 'No action items found.'
    """
  )

  reduced_transcript = reduce_transcript(map_chain, transcript, "No action items found.")

  if reduced_transcript == "No action items found.":
    return reduced_transcript

  return reduce_chain.invoke({"text" : reduced_transcript})

def extract_key_decisions(transcript: str) -> str:

  map_chain = build_chain(
    """
    You are an expert meeting analyst.

    Extract all key decisions made in this portion of the meeting transcript.

    Return a numbered list.

    If no decisions were made, return:
    'No key decisions found.'
    """
  )

  reduce_chain = build_chain(
    """
    You are an expert meeting analyst.

    You are given key decisions extracted from different portions of the same meeting.

    Merge duplicate decisions.
    Combine partial information.
    Return a clean numbered list.

    If no decisions exist, return:
    'No key decisions found.'
    """
  )

  reduced_transcript = reduce_transcript(map_chain, transcript, "No key decisions found.")

  if reduced_transcript == "No key decisions found.":
    return reduced_transcript

  return reduce_chain.invoke({"text": reduced_transcript})


def extract_questions(transcript: str) -> str:

  map_chain = build_chain(
    """
    You are an expert meeting analyst.

    Extract only questions, uncertainties, or follow-up items that were explicitly mentioned in this portion of the transcript. 
    Do not invent or infer new questions.

    Return a numbered list.

    If none exist, return:
    'No open questions found.'
    """
  )

  reduce_chain = build_chain(
    """
    You are an expert meeting analyst.

    You are given unresolved questions extracted from different portions of the same meeting.

    Merge duplicate questions.
    Combine partial information.
    Return a clean numbered list.

    If no open questions exist, return:
    'No open questions found.'
    """
  )

  reduced_transcript = reduce_transcript(map_chain, transcript, "No open questions found.")  

  if reduced_transcript == "No open questions found.":
    return reduced_transcript

  return reduce_chain.invoke({"text": reduced_transcript})