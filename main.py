from dotenv import load_dotenv
load_dotenv()
from vector_store import build_vector_store
from transcriber import transcribe_all
from audio_processor import process_input
from summarizer import generate_summary,generate_title
from extractor import extract_action_items,extract_key_decisions,extract_questions
from rag_engine import build_rag_chain, ask_question
from rich import print
from pathlib import Path

def ai_video_assistant_pipeline(source :str, language :str = "english"):
  

  chunks = process_input(source)

  transcript = transcribe_all(chunks=chunks,language=language)
  
  build_vector_store(transcript)

  summary = generate_summary(transcript)

  title = generate_title(summary)

  actions = extract_action_items(transcript)
  decisions = extract_key_decisions(transcript)
  questions = extract_questions(transcript)
  rag_chain = build_rag_chain()


  return {
    "summary" : summary,
    "title" : title,
    "actions" : actions,
    "decisions" : decisions,
    "questions" : questions,
    "rag_chain" : rag_chain
  }

if __name__ == "__main__":

  source = input("Enter YouTube URL or local file path: ").strip()
  language = input("Language (english/hinglish): ").strip() or "english"
  result = ai_video_assistant_pipeline(source, language)

  print("\n","="*60,"\n", result['title'],"\n")

  print("\n","="*60,"\n", result['summary'],"\n")

  print("\n","="*60,"\n", result['actions'],"\n")
  print("\n","="*60,"\n", result['decisions'],"\n")
  print("\n","="*60,"\n", result['questions'],"\n")

  
  print("Press 0 to exit.\n")
  print("Any questions? \n")

  while True:
    message = input("\n-> ")

    if message == '0':
      break

    response = ask_question(result['rag_chain'], message)

    print(response)