# query_api/main.py
from fastapi import FastAPI, Body
import os
from langchain_groq import GroqEmbeddings, ChatGroq
from langchain_postgres import PGVector
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

app = FastAPI()

# Env vars
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NEON_POSTGRES_URI = os.getenv("NEON_POSTGRES_URI")

# Groq Embeddings
embeddings = GroqEmbeddings(
    model="nomic-embed-text",
    groq_api_key=GROQ_API_KEY
)

# Neon vector DB
vectorstore = PGVector(
    embeddings=embeddings,
    connection=NEON_POSTGRES_URI,
    collection_name="resumes",
    use_jsonb=True,
)
retriever = vectorstore.as_retriever()

# Groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",  # or llama2-70b-4096
    temperature=0.3,
    groq_api_key=GROQ_API_KEY,
)

# RAG Chain
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

@app.post("/query")
def query_rag(question: str = Body(..., embed=True)):
    """
    Query Neon embeddings with Groq RAG.
    """
    try:
        answer = rag_chain.invoke(question)
        return {"question": question, "answer": answer}
    except Exception as e:
        return {"status": "error", "message": str(e)}
