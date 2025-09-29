from fastapi import FastAPI, Body
from langchain.schema import Document
from langchain_postgres import PGVector
from langchain_groq import ChatGroq
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
import uuid
from google import genai

app = FastAPI()

# =========================
# Environment variables
# =========================
NEON_POSTGRES_URI = os.getenv("NEON_POSTGRES_URI")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# =========================
# Gemini client
# =========================
client = genai.Client(api_key=GEMINI_API_KEY)

# =========================
# Safe embed_text for latest Gemini SDK
# =========================
def embed_text(text: str):
    if not isinstance(text, str):
        raise ValueError("embed_text only accepts strings, not dicts or lists of dicts")
    
    response = client.models.embed_content(
        model="text-embedding-004",
        contents=[text]  # must be a list
    )
    
    if not response.embeddings or not response.embeddings[0].values:
        raise ValueError("Gemini API returned no embedding")
    
    return response.embeddings[0].values

# =========================
# Gemini embeddings wrapper
# =========================
class GeminiEmbeddings:
    def embed_documents(self, texts):
        return [embed_text(str(t)) for t in texts]

    def embed_query(self, text):
        return embed_text(str(text))

embeddings = GeminiEmbeddings()

# =========================
# PGVector (Neon Postgres)
# =========================
vectorstore = PGVector(
    embeddings=embeddings,
    connection=NEON_POSTGRES_URI,
    collection_name="resumes",
    use_jsonb=True,
)

# =========================
# Groq LLM for RAG
# =========================
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.3,
    groq_api_key=GROQ_API_KEY,
)

# =========================
# RAG prompt
# =========================
prompt = hub.pull("rlm/rag-prompt")

# =========================
# Helper functions
# =========================
def format_docs(docs):
    return "\n".join(doc.page_content if hasattr(doc, "page_content") else str(doc) for doc in docs)

def get_context(question: str, user_id: str):
    """Fetch relevant documents only for a specific user"""
    retriever = vectorstore.as_retriever(search_kwargs={"filter": {"user_id": user_id}})
    docs = retriever.get_relevant_documents(question)
    return format_docs(docs)

# =========================
# API Endpoints
# =========================
@app.post("/ingest")
def ingest_resume(resume_text: str = Body(...), user_id: str = Body(...)):
    """Store a resume in Neon with embeddings, tied to a user_id"""
    try:
        doc_id = str(uuid.uuid4())
        docs = [
            Document(
                page_content=resume_text,
                metadata={"source": "resume", "id": doc_id, "user_id": user_id}
            )
        ]
        vectorstore.add_documents(docs)
        return {"status": "success", "message": "Resume stored", "id": doc_id, "user_id": user_id}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/query")
def query_rag(question: str = Body(...), user_id: str = Body(...)):
    """Query only the resumes belonging to a specific user_id"""
    try:
        context = get_context(question, user_id)
        rag_chain = (
            {"context": lambda _: context, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        answer = rag_chain.invoke({"question": question})
        return {"user_id": user_id, "question": question, "answer": answer}
    except Exception as e:
        return {"status": "error", "message": str(e)}
