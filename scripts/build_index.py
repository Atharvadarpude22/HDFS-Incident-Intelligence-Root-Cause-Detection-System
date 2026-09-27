"""knowledge_base/*.md -> loader -> chunker -> embeddings -> FAISS -> vector_db/

Usage: python scripts/build_index.py [--kb data/knowledge_base] [--out vector_db]
                                     [--query "DataNode heartbeat failure"]
"""
import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # project root
sys.path.insert(0, str(ROOT))

from src.rag import DocumentLoader, DocumentChunker, EmbeddingService, VectorStore, Retriever

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("build_index")


def build_index(kb_path, index_path, chunk_size=500, overlap=50, embedding_service=None):
    documents = DocumentLoader(kb_path).load_documents()
    chunks = DocumentChunker(chunk_size=chunk_size, overlap=overlap).chunk_documents(documents)
    if not chunks:
        raise ValueError("No chunks were produced from the knowledge base.")

    store = VectorStore(embedding_service or EmbeddingService())
    store.build(chunks)
    store.save(index_path)
    logger.info("Indexed %d chunks from %d documents into %s", len(chunks), len(documents), index_path)
    return store, chunks


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--kb", default=str(ROOT / "data" / "knowledge_base"))
    p.add_argument("--out", default=str(ROOT / "vector_db"))
    p.add_argument("--chunk-size", type=int, default=500)
    p.add_argument("--overlap", type=int, default=50)
    p.add_argument("--query", help="optional: run a test search after building")
    a = p.parse_args()

    embedding_service = EmbeddingService()
    store, chunks = build_index(a.kb, a.out, a.chunk_size, a.overlap, embedding_service)

    if a.query:  # helps you tune max_distance / min_bm25_score
        retriever = Retriever(embedding_service, store, max_distance=10, min_bm25_score=0).build_sparse_index(chunks)
        for r in retriever.retrieve_with_scores(a.query, top_k=5):
            r["content"] = r["content"][:100]
            print(json.dumps(r, indent=1))