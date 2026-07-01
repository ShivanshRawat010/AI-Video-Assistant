from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel, Runnable
from vector_store import get_retriever
from dotenv import load_dotenv
load_dotenv()

def load_model():
  return ChatMistralAI(temperature=0)

def format_docs(docs):
  return "\n".join(doc.page_content  for doc in docs)

def build_rag_chain():
  
  llm = load_model()

  prompt = ChatPromptTemplate.from_messages(
    [(
      "system",
      """You are an expert meeting assistant. Answer the user's question 
      based ONLY on the meeting transcript context provided below.

      If the answer is not found in the context, say: 
      "I could not find this information in the meeting transcript."

      Always be concise and precise. If quoting someone, mention it clearly.

      Context from meeting transcript:
      {context}""",
    ),
      ("human", "{question}"),
    ]
  )

  retriever = get_retriever()

  chain = (
    {
      "question" : RunnablePassthrough(),
      "context" : retriever | RunnableLambda(format_docs)
    }
    | prompt
    | llm
    | StrOutputParser()
  )

  return chain

def ask_question(rag_chain : Runnable, question : str):
  return rag_chain.invoke(question)