# dv_rag.py
# Reusable RAG helper for the DV Helpdesk Agent.
# Knowledge source: rag_knowledge.db
#
# Expected SQLite table:
# chunks(chunk_id, page_number, source, chunk_text)

import sqlite3
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings

RAG_DB = "rag_knowledge.db"
EMBED_MODEL = "nomic-embed-text"

_vector_store = None
_indexed_db = None


def _load_chunks(db_file):
    """Read the preprocessed policy chunks stored in SQLite."""
    if not Path(db_file).exists():
        raise FileNotFoundError(f"RAG database not found: {db_file}")

    conn = sqlite3.connect(db_file)
    rows = conn.execute("""
        SELECT chunk_id, page_number, source, chunk_text
        FROM chunks
        ORDER BY chunk_id
    """).fetchall()
    conn.close()

    return [
        Document(
            page_content=row[3],
            metadata={"chunk_id": row[0], "page": row[1], "source": row[2]}
        )
        for row in rows
    ]


def build_rag_index(db_file=RAG_DB, embedding_model=EMBED_MODEL):
    """Build the in-memory vector index once and reuse it for later queries."""
    global _vector_store, _indexed_db

    db_path = str(Path(db_file).resolve())
    if _vector_store is not None and _indexed_db == db_path:
        return _vector_store

    documents = _load_chunks(db_file)
    embeddings = OllamaEmbeddings(model=embedding_model)
    _vector_store = InMemoryVectorStore(embedding=embeddings)
    _vector_store.add_documents(documents)
    _indexed_db = db_path

    return _vector_store


def retrieve_dv_requirements(dv_type, question, analyst_remarks="", k=3,
                             db_file=RAG_DB, embedding_model=EMBED_MODEL):
    """Retrieve the most relevant checklist chunks for a DV inquiry."""
    store = build_rag_index(db_file, embedding_model)

    query = (
        f"DV or transaction type: {dv_type}\n"
        f"Accounting Analyst remarks: {analyst_remarks}\n"
        f"User question: {question}\n"
        "Find the most relevant supporting documents, attachment requirements, "
        "and conditions from the pre-audit checklist."
    )

    docs = store.similarity_search(query, k=k)

    return [
        {
            "chunk_id": doc.metadata.get("chunk_id"),
            "page": doc.metadata.get("page"),
            "source": doc.metadata.get("source"),
            "text": doc.page_content
        }
        for doc in docs
    ]


def format_evidence(evidence):
    """Convert retrieved chunks into readable evidence for an LLM prompt."""
    return "\n\n".join(
        f"[{item['chunk_id']} | Page {item['page']}]\n{item['text']}"
        for item in evidence
    )
