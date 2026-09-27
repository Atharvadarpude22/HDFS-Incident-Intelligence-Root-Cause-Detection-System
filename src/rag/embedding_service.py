from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingService:
    """HuggingFace sentence-embedding wrapper (LangChain-compatible)."""

    def __init__(self, model_name="BAAI/bge-small-en-v1.5", device="cpu"):
        self.model_name = model_name
        self.model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
        )

    def embed_query(self, query):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")
        return self.model.embed_query(query)

    def embed_documents(self, texts):
        return self.model.embed_documents(texts)

    def get_embedding_dimension(self):
        return len(self.model.embed_query("dimension check"))