import logging
import re
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Heading-based split first, then size-based split for large sections."""

    def __init__(self, chunk_size=500, overlap=50, min_chunk_chars=20):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size.")
        self.min_chunk_chars = min_chunk_chars
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
            strip_headers=False,  # keep headings inside the chunk text
        )
        self.size_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=overlap
        )
        self.chunks = []

    @staticmethod
    def clean_text(text):
        text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
        text = re.sub(r"[ ]+$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*---+\s*$", "", text, flags=re.MULTILINE)
        return re.sub(r"\n{3,}", "\n\n", text).strip()

    def chunk_documents(self, documents):
        chunks = []
        for doc in documents:
            cleaned = self.clean_text(doc.page_content or "")
            if not cleaned:
                logger.warning("Skipping empty document: %s", doc.metadata.get("source"))
                continue

            sections = self.header_splitter.split_text(cleaned)
            for s in sections:
                s.metadata.update(doc.metadata)  # carry source/path forward

            chunks.extend(self.size_splitter.split_documents(sections))

        # Drop heading-only / near-empty chunks, then assign global ids
        chunks = [c for c in chunks if len(c.page_content.strip()) >= self.min_chunk_chars]
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i

        self.chunks = chunks
        return chunks

    def get_chunk_count(self):
        return len(self.chunks)