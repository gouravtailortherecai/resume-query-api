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

def embed_text(text: str):
    """Call Gemini embeddings and return vector"""
    emb_response = client.models.embed_content(
        model="text-embedding-004",
        contents=text
    )
    # Extract embedding from the response
    return emb_response.data[0].embedding

class GeminiEmbeddings:
    def embed_documents(self, texts):
        return [embed_text(t) for t in texts]

    def embed_query(self, text):
        return embed_text(text)

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
# Utility to format retrieved documents
# =========================
def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)

def get_context(question: str):
    """Fetch relevant documents and format them as context"""
    docs = retriever.get_relevant_documents(question)
    return format_docs(docs)

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
        # Pass a dictionary matching chain inputs
        answer = rag_chain.invoke({"question": question})
        return {"question": question, "answer": answer}
    except Exception as e:
        return {"status": "error", "message": str(e)}
