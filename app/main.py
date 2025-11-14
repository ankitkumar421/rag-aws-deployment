from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from app.rag_utils import load_text_file, split_docs, create_or_load_index, build_retriever

# === Initialize FastAPI app ===
app = FastAPI(title="RAG Microservice (In-Memory)", version="1.0")

# Global variables for in-memory state
INDEX = None
RETRIEVER = None
SAMPLE_DOC_PATH = Path("app/sample_docs/sample.txt")

# === Pydantic models ===
class QueryRequest(BaseModel):
    query: str
    k: int = 3

class IngestResponse(BaseModel):
    message: str
    chunks_indexed: int

class QueryResponse(BaseModel):
    query: str
    top_chunks: list[str]

# === Routes ===
@app.post("/ingest", response_model=IngestResponse)
def ingest_docs():
    """
    Load documents from app/sample_docs/, split them into chunks,
    and build an in-memory index.
    """
    global INDEX, RETRIEVER

    if not SAMPLE_DOC_PATH.exists():
        raise HTTPException(status_code=404, detail="No sample file found in app/sample_docs/")

    docs = load_text_file(str(SAMPLE_DOC_PATH))
    chunks = split_docs(docs)
    INDEX = create_or_load_index(chunks)
    RETRIEVER = build_retriever(INDEX)

    return {"message": "Index built successfully!", "chunks_indexed": len(chunks)}

@app.post("/query", response_model=QueryResponse)
def query_docs(request: QueryRequest):
    """
    Search the in-memory index for relevant chunks.
    """
    global RETRIEVER
    if RETRIEVER is None:
        raise HTTPException(status_code=400, detail="No index loaded. Run /ingest first.")

    results = RETRIEVER.get_relevant_documents(request.query, k=request.k)
    top_chunks = [r.page_content[:200] for r in results]

    return {"query": request.query, "top_chunks": top_chunks}

@app.get("/")
def root():
    return {"message": "RAG microservice is running. Visit /docs for API usage."}
