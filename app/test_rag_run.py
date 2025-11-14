from pathlib import Path
from app.rag_utils import load_text_file, split_docs, create_or_load_index, build_retriever

sample_path = Path("app/sample_docs/sample.txt")
print("Sample path:", sample_path.resolve())

docs = load_text_file(str(sample_path))
print(f"Loaded {len(docs)} document(s).")

chunks = split_docs(docs)
print(f"Split into {len(chunks)} chunk(s).")

idx = create_or_load_index(chunks, persist_dir=None)
print("In-memory index created.")

retriever = build_retriever(idx)
print("Running retrieval...")

results = retriever.get_relevant_documents("What is RAG and why is it useful?", k=2)
print("\nRetrieved chunks:")
for r in results:
    print("-", r.page_content[:200], "...")
