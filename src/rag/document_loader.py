import logging
from pathlib import Path
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Loads Markdown troubleshooting docs (including subfolders) as Documents."""

    def __init__(self, knowledge_base_path):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.documents = []
        self.skipped = []

    def load_documents(self):
        if not self.knowledge_base_path.exists():
            raise FileNotFoundError(f"Knowledge base not found: {self.knowledge_base_path}")
        if not self.knowledge_base_path.is_dir():
            raise NotADirectoryError(f"Expected a directory: {self.knowledge_base_path}")

        self.documents, self.skipped = [], []

        for path in sorted(self.knowledge_base_path.rglob("*.md")):  # recursive
            try:
                try:
                    content = path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    logger.warning("Non-UTF-8 file, decoding with replacement: %s", path)
                    content = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                logger.warning("Skipping unreadable file %s: %s", path, exc)
                self.skipped.append((str(path), f"unreadable: {exc}"))
                continue

            if not content.strip():
                logger.warning("Skipping empty file: %s", path)
                self.skipped.append((str(path), "empty"))
                continue

            # Relative path keeps sources unique across subfolders
            source = path.relative_to(self.knowledge_base_path).as_posix()
            self.documents.append(
                Document(page_content=content, metadata={"source": source, "path": str(path)})
            )

        if not self.documents:
            raise ValueError(f"No usable .md documents found in {self.knowledge_base_path}")
        return self.documents

    def get_document_count(self):
        return len(self.documents)

    def get_document_names(self):
        return [d.metadata["source"] for d in self.documents]