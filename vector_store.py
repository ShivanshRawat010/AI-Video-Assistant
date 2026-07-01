from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from inspect import signature
import shutil
from pathlib import Path

CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"

model = None

def get_embedding():
  global model

  if model == None:
    model = HuggingFaceEmbeddings(
      model_name = EMBEDDING_MODEL
    )

  return model

def build_vector_store(transcript : str):

  db_path = Path("./vector_db")

  if db_path.exists():
    shutil.rmtree(db_path)

  splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
  )

  chunks = splitter.split_text(transcript)

  docs = [Document(page_content=chunk, metadata = {"chunk_index" : i}) for i,chunk in enumerate(chunks)]

  embedding = get_embedding()

  vector_store = Chroma.from_documents(
    embedding=embedding,
    documents=docs,
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_DIR
  )

  return vector_store


def load_vector_store():

  embedding_function = get_embedding()

  return Chroma(
    embedding_function=embedding_function,
    persist_directory=CHROMA_DIR,
    collection_name=COLLECTION_NAME
  )


def get_retriever():
  vector_store = load_vector_store()

  retriever = vector_store.as_retriever(
    search_type = 'mmr',
    search_kwargs = {
      'k' : 3,
      'fetch_k' : 10,
      'lambda_mult' : 0.7
    }
  )

  return retriever