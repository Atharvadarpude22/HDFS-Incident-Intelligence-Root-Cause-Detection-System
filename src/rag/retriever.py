import re

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

try:
    # LangChain 1.x
    from langchain_classic.retrievers import EnsembleRetriever
except ImportError:
    # Older LangChain versions
    from langchain.retrievers import EnsembleRetriever


def tokenize(text):
    """
    Lowercase text and preserve dotted/dashed tokens such as:
    dfs.datanode.du.reserved
    """
    tokens = (
        t.strip(".-")
        for t in re.findall(r"[\w.\-]+", str(text).lower())
    )
    return [t for t in tokens if t]


class Retriever:
    """
    Hybrid retriever using:

    1. FAISS dense retrieval for semantic similarity
    2. BM25 sparse retrieval for keyword matching
    3. EnsembleRetriever for fusion
    4. Relevance cutoff using either dense distance OR BM25 score
    """

    def __init__(
        self,
        vector_store,
        dense_weight=0.7,
        sparse_weight=0.3,
        max_distance=0.9,
        min_bm25_score=2.0,
    ):
        if vector_store is None:
            raise ValueError("vector_store is required.")

        self.vector_store = vector_store

        self.weights = [
            dense_weight,
            sparse_weight,
        ]

        self.max_distance = max_distance
        self.min_bm25_score = min_bm25_score

        self.bm25 = None
        self.hybrid = None

        self.candidate_k = 10
        self.last_query = None
        self.last_results = []

    def build_sparse_index(self, chunks=None, candidate_k=10):
        """
        Build BM25 and the hybrid retriever.

        When chunks=None, rebuild BM25 from documents stored in FAISS.
        This is required after loading a saved vector index.
        """

        if chunks is None:
            chunks = self.vector_store.get_documents()

        if not chunks:
            raise ValueError("Chunks cannot be empty.")

        self.candidate_k = candidate_k

        self.bm25 = BM25Retriever.from_documents(
            chunks,
            preprocess_func=tokenize,
            k=candidate_k,
        )

        dense_retriever = self.vector_store.as_retriever(
            candidate_k
        )

        self.hybrid = EnsembleRetriever(
            retrievers=[
                dense_retriever,
                self.bm25,
            ],
            weights=self.weights,
        )

        return self

    @staticmethod
    def _key(doc):
        """
        Generate a stable key for matching documents returned by
        dense retrieval and BM25.
        """
        chunk_id = doc.metadata.get("chunk_id")

        if chunk_id is not None:
            return (
                doc.metadata.get("source"),
                chunk_id,
            )

        return doc.page_content

    def retrieve(self, query, top_k=3):
        """
        Retrieve relevant documents.

        A document is retained when either:

            dense_distance <= max_distance

        OR

            bm25_score >= min_bm25_score

        Returned documents contain:

            dense_distance
            bm25_score
        """

        # -------------------------
        # Validate query
        # -------------------------
        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or top_k < 1
        ):
            raise ValueError("top_k must be a positive integer.")

        if self.hybrid is None:
            raise ValueError(
                "Call build_sparse_index() first."
            )

        tokens = tokenize(query)

        if not tokens:
            raise ValueError(
                "Query has no searchable terms."
            )

        self.last_query = query

        # -------------------------
        # Hybrid retrieval
        # -------------------------
        fused_docs = self.hybrid.invoke(query)

        # -------------------------
        # Dense scores
        # -------------------------
        dense_results = self.vector_store.search(
            query,
            self.candidate_k,
        )

        dense = {
            self._key(doc): float(distance)
            for doc, distance in dense_results
        }

        # -------------------------
        # BM25 scores
        # -------------------------
        bm25_scores = self.bm25.vectorizer.get_scores(tokens)

        sparse = {
            self._key(doc): float(score)
            for doc, score in zip(
                self.bm25.docs,
                bm25_scores,
            )
        }

        # -------------------------
        # Apply relevance cutoff
        # -------------------------
        results = []

        for doc in fused_docs:
            key = self._key(doc)

            dense_distance = dense.get(key)
            bm25_score = sparse.get(key, 0.0)

            dense_match = (
                dense_distance is not None
                and dense_distance <= self.max_distance
            )

            sparse_match = (
                bm25_score >= self.min_bm25_score
            )

            if dense_match or sparse_match:
                results.append(
                    Document(
                        page_content=doc.page_content,
                        metadata={
                            **doc.metadata,
                            "dense_distance": dense_distance,
                            "bm25_score": bm25_score,
                        },
                    )
                )

        self.last_results = results[:top_k]

        return self.last_results

    def retrieve_with_scores(self, query, top_k=3):
        """
        Same retrieval as retrieve(), but returns plain dictionaries.
        Useful for APIs and debugging.
        """

        return [
            {
                "rank": rank,
                "source": doc.metadata.get(
                    "source",
                    "unknown",
                ),
                "chunk_id": doc.metadata.get(
                    "chunk_id"
                ),
                "dense_distance": doc.metadata.get(
                    "dense_distance"
                ),
                "bm25_score": doc.metadata.get(
                    "bm25_score"
                ),
                "content": doc.page_content,
            }
            for rank, doc in enumerate(
                self.retrieve(query, top_k=top_k),
                start=1,
            )
        ]

    def get_retrieval_stats(self):
        """
        Return statistics for the most recent retrieval.
        """

        return {
            "query": self.last_query,
            "number_of_results": len(
                self.last_results
            ),
            "sources": [
                doc.metadata.get("source")
                for doc in self.last_results
            ],
        }


def load_retriever(
    index_path,
    embedding_service=None,
    **retriever_kwargs,
):
    """
    Load a saved FAISS index and rebuild BM25.

    Returns a ready-to-use Retriever.
    """

    from pathlib import Path

    index_path = Path(index_path)

    if not (index_path / "index.faiss").exists():
        raise FileNotFoundError(
            f"FAISS index not found at {index_path}. "
            "Run scripts/build_index.py first."
        )

    from .embedding_service import EmbeddingService
    from .vector_store import VectorStore

    embedding_service = (
        embedding_service
        or EmbeddingService()
    )

    store = VectorStore(
        embedding_service
    ).load(index_path)

    return (
        Retriever(
            store,
            **retriever_kwargs,
        )
        .build_sparse_index()
    )