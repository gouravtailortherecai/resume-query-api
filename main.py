from fastapi import FastAPI, Body
import os
from google import genai
from langchain_postgres import PGVector
from langchain_groq import ChatGroq
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

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
# Safe embed_text function
# =========================
def embed_text(text: str):
    """Embed a plain string with Gemini embeddings API"""
    if not isinstance(text, str):
        raise ValueError("embed_text only accepts strings, not dicts or lists of dicts")
    emb_response = client.models.embed_content(
        model="text-embedding-004",
        contents=[text]  # wrap single string in a list
    )
    return emb_response.data[0].embedding

# =========================
# Gemini embeddings wrapper
# =========================
class GeminiEmbeddings:
    def embed_documents(self, texts):
        # Ensure every element is a string
        return [embed_text(t if isinstance(t, str) else str(t)) for t in texts]

    def embed_query(self, text):
        return embed_text(str(text))  # convert to string if needed

embeddings = GeminiEmbeddings()

# =========================
# PGVector (Neon Postgres) Setup
# =========================
vectorstore = PGVector(
    embeddings=embeddings,
    connection=NEON_POSTGRES_URI,
    collection_name="resumes",
    use_jsonb=True,
)

retriever = vectorstore.as_retriever()

# =========================
# Groq LLM for answers
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
# Utilities
# =========================
def format_docs(docs):
    """Combine retrieved docs into a single string"""
    return "\n".join(doc.page_content if hasattr(doc, "page_content") else str(doc) for doc in docs)

def get_context(question: str):
    """Fetch relevant documents from vector store"""
    docs = retriever.get_relevant_documents(question)
    return format_docs(docs)  # always returns a string

# =========================
# RAG chain
# =========================
rag_chain = (
    {"context": get_context, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# =========================
# API Endpoint
# =========================
@app.post("/query")
def query_rag(question: str = Body(..., embed=True)):
    try:
        # Pass dictionary to RAG chain, embeddings get only strings internally
        answer = rag_chain.invoke({"question": question})
        return {"question": question, "answer": answer}
    except Exception as e:
        return {"status": "error", "message": str(e)}
