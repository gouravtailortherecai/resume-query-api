# main.py (Query API)
from fastapi import FastAPI, Body
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_groq import ChatGroq
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

app = FastAPI()

# Env vars
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NEON_POSTGRES_URI = os.getenv("NEON_POSTGRES_URI")

# Embeddings + Vector DB
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = PGVector(
    embeddings=embeddings,
    connection=NEON_POSTGRES_URI,
    collection_name="resumes",
    use_jsonb=True,
)
retriever = vectorstore.as_retriever()

# LLM
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.3)

# RAG chain
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
    try:
        answer = rag_chain.invoke(question)
        return {"question": question, "answer": answer}
    except Exception as e:
        return {"status": "error", "message": str(e)}
