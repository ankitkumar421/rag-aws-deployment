import os
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# === CONFIG ===
MODEL_NAME = os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")  # keep CPU for WSL
EMBEDDING_DIR = Path("/tmp/embeddings_cache")  # optional persistence (not used in this simple impl)

# Initialize sentence-transformers model (CPU)
_model = SentenceTransformer(MODEL_NAME, device=EMBEDDING_DEVICE)

def _embed_texts(texts: List[str]) -> np.ndarray:
    """Return numpy array of embeddings shape (n, d) and always 2-D."""
    embs = _model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    embs = np.asarray(embs, dtype=float)
    if embs.ndim == 1:
        embs = np.expand_dims(embs, 0)
    return embs

# Simple in-memory index structure
class SimpleInMemoryIndex:
    def __init__(self, texts: List[str], metadatas: List[Dict[str, Any]] = None):
        self.texts = texts
        self.metadatas = metadatas or [{} for _ in texts]
        self.embeddings = _embed_texts(texts)  # shape (n, d)
        # Ensure 2-D
        self.embeddings = np.atleast_2d(self.embeddings)
        # Normalize rows for cosine similarity, avoid divide-by-zero
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.normed = self.embeddings / norms

    def query(self, qtext: str, top_k: int = 3):
        q_emb = _embed_texts([qtext])  # (1, d)
        q_emb = np.atleast_2d(q_emb)
        qnorm = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)
        sims = (self.normed @ qnorm.T).squeeze()
        # Ensure sims is 1-D array even if only one doc
        sims = np.asarray(sims)
        if sims.ndim == 0:
            sims = np.expand_dims(sims, 0)
        top_idx = np.argsort(-sims)[:top_k]
        results = []
        for i in top_idx:
            results.append({"text": self.texts[int(i)], "score": float(sims[int(i)]), "metadata": self.metadatas[int(i)]})
        return results

# === Utility functions used by test script ===
def load_text_file(path: str):
    """Load a .txt file into a list-of-documents (LangChain Document objects)."""
    loader = TextLoader(path)
    return loader.load()

def split_docs(docs, chunk_size=800, overlap=100):
    """Return list of Document-like objects (LangChain Document) chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.split_documents(docs)
    return chunks

def create_or_load_index(chunks, persist_dir: str = None):
    """
    Build a SimpleInMemoryIndex from list of langchain Document chunks.
    persist_dir argument is accepted but ignored in this simple impl.
    """
    texts = [c.page_content for c in chunks]
    metadatas = [getattr(c, "metadata", {}) for c in chunks]
    idx = SimpleInMemoryIndex(texts, metadatas)
    return idx

def build_retriever(index):
    """
    Return a simple retriever object with a `get_relevant_documents(query)` method
    to be consistent with earlier tests. We adapt the returned format.
    """
    class Retriever:
        def __init__(self, index):
            self.index = index
        def get_relevant_documents(self, query, k=3):
            docs = self.index.query(query, top_k=k)
            # convert to a simple object with page_content (to match previous expectations)
            class D:
                def __init__(self, txt):
                    self.page_content = txt
            return [D(d["text"]) for d in docs]
    return Retriever(index)
