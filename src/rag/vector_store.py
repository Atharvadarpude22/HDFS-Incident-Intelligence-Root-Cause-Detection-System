from langchain_community.vectorstores import FAISS


class VectorStore:
    """FAISS store via LangChain; documents + metadata are saved with the index."""

    def __init__(self, embedding_service):
        self.embeddings = embedding_service.model
        self.db = None

    def build(self, chunks):
        if not chunks:
            raise ValueError("Chunks cannot be empty (no usable content was found).")
        self.db = FAISS.from_documents(chunks, self.embeddings)
        return self.db

    def search(self, query, top_k=3):
        """Returns [(Document, squared-L2 distance)], lower = more similar."""
        self._check_ready()
        top_k = min(top_k, self.get_size())
        return [
            (doc, float(dist))
            for doc, dist in self.db.similarity_search_with_score(query, k=top_k)
        ]

    def as_retriever(self, top_k=5):
        self._check_ready()
        return self.db.as_retriever(search_kwargs={"k": min(top_k, self.get_size())})

    def get_documents(self):
        """All stored chunks in index order (used to rebuild BM25 after load())."""
        self._check_ready()
        return [
            self.db.docstore.search(self.db.index_to_docstore_id[i])
            for i in range(self.db.index.ntotal)
        ]

    def save(self, path="it-incident-intelligence-agent\vector_db"):
        self._check_ready()
        self.db.save_local(path)

    def load(self, path):
        # Only load indexes you created yourself (pickle-based metadata).
        self.db = FAISS.load_local(
            path, self.embeddings, allow_dangerous_deserialization=True
        )
        return self

    def get_size(self):
        return self.db.index.ntotal if self.db else 0

    def is_ready(self):
        return self.db is not None and self.db.index.ntotal > 0

    def _check_ready(self):
        if not self.is_ready():
            raise ValueError("Vector store has not been built or loaded.")