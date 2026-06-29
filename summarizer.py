from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

def load_model():
  return ChatMistralAI() 

def split_transcript(transcript : str) -> list:
  
  splitter = RecursiveCharacterTextSplitter(
    chunk_size = 3000,
    chunk_overlap = 200
  )

  return splitter.split_text(transcript)

def generate_summary(transcript : str) -> str:

  llm = load_model()
  chunks = split_transcript(transcript)
  parser = StrOutputParser()

  chunk_summary_prompt = ChatPromptTemplate.from_messages(
    [
      ("system", "Summarize this portion of a meeting transcript concisely."),
      ("human", "{text}"),
    ]
  )

  chunk_summary_chain = chunk_summary_prompt | llm | parser

  chunk_summaries = [chunk_summary_chain.invoke({"text" : chunk}) for chunk in chunks]

  combined = "\n\n".join(chunk_summaries)

  combined_prompt = ChatPromptTemplate.from_messages(
    [
      (
        "system",
        "You are an expert meeting summarizer. Combine these partial summaries "
        "into one final professional meeting summary in bullet points.",
      ),
      ("human", "{text}"),
    ]
  )

  final_chain = combined_prompt | llm | parser

  return final_chain.invoke({"text" : combined})

def generate_title(summary : str) -> str:

  llm = load_model()

  title_prompt = ChatPromptTemplate.from_messages([
    (
      "system",
      "Based on the meeting summary, generate a short professional meeting title "
      "(max 8 words). Only return the title, nothing else.",
    ),
    ("human", "{text}"),
  ])

  parser = StrOutputParser()

  title_chain = title_prompt | llm | parser

  return title_chain.invoke({"text" : summary[:2000]})