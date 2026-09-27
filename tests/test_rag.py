import sys
from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.document_loader import DocumentLoader
from src.rag.document_chunker import DocumentChunker
from src.rag.vector_store import VectorStore
from src.rag.retriever import Retriever
from src.rag.rag_pipeline import RAGPipeline, LLMGenerationError, NO_CONTEXT_MESSAGE


class FakeEmbeddingService:
    model = DeterministicFakeEmbedding(size=64)


@pytest.fixture
def kb(tmp_path):
    (tmp_path / "disk.md").write_text(
        "# DataNode disk full\n\nSymptom: No space left on device in datanode logs.\n"
        "Check dfs.datanode.du.reserved and run df -h on each volume.\n"
        "## Action\nFree disk space or add volumes, then restart the DataNode.\n")
    (tmp_path / "namenode.md").write_text(
        "# NameNode safe mode\n\nNameNode stays in safe mode when block reports are missing.\n"
        "Run hdfs dfsadmin -safemode get to check the state.\n")
    (tmp_path / "empty.md").write_text("   \n")
    (tmp_path / "bad.md").write_bytes(b"# Bad encoding\n\nCaf\xe9 log line with enough text here.\n")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "nested.md").write_text("# Nested\n\nBlock corruption detected by fsck in nested docs.\n")
    return tmp_path


def build(kb, max_distance=1e9, min_bm25=2.0):
    docs = DocumentLoader(kb).load_documents()
    chunks = DocumentChunker(chunk_size=200, overlap=20).chunk_documents(docs)
    store = VectorStore(FakeEmbeddingService())
    store.build(chunks)
    retr = Retriever(store, max_distance=max_distance, min_bm25_score=min_bm25)
    retr.build_sparse_index(chunks)
    return docs, chunks, store, retr


def test_loader_edge_cases(kb):
    loader = DocumentLoader(kb)
    docs = loader.load_documents()
    names = loader.get_document_names()
    assert "sub/nested.md" in names            # recursive
    assert "bad.md" in names                   # non-UTF-8 handled
    assert "empty.md" not in names             # blank skipped
    assert len(loader.skipped) == 1


def test_loader_missing_and_empty_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        DocumentLoader(tmp_path / "nope").load_documents()
    with pytest.raises(ValueError):
        DocumentLoader(tmp_path).load_documents()


def test_chunker_metadata(kb):
    docs, chunks, *_ = build(kb)
    assert chunks and all("source" in c.metadata for c in chunks)
    assert [c.metadata["chunk_id"] for c in chunks] == list(range(len(chunks)))


def test_query_validation(kb):
    *_, retr = build(kb)
    for bad in ["", "   ", None, 123, "!!!"]:
        with pytest.raises((ValueError, TypeError)):
            retr.retrieve(bad)


def test_keyword_match_is_case_insensitive(kb):
    *_, retr = build(kb, max_distance=0.0)     # dense rejects all -> BM25 only
    res = retr.retrieve("DFS.DataNode.DU.Reserved no space left", top_k=2)
    assert res and res[0].metadata["source"] == "disk.md"


def test_relevance_cutoff_returns_nothing(kb):
    *_, store, retr = build(kb, max_distance=0.0, min_bm25=1e9)
    assert retr.retrieve("weather in paris") == []


def test_load_then_rebuild_bm25(kb, tmp_path):
    _, _, store, _ = build(kb)
    store.save(tmp_path / "idx")
    store2 = VectorStore(FakeEmbeddingService()).load(tmp_path / "idx")
    retr2 = Retriever(store2, max_distance=1e9)
    retr2.build_sparse_index()                 # no chunks passed
    assert retr2.retrieve("safe mode block reports")


def test_pipeline_answer_and_evidence(kb):
    *_, retr = build(kb)
    rag = RAGPipeline(retr, FakeListChatModel(responses=["root cause: disk full"]))
    assert "disk full" in rag.answer("No space left on device")
    ev = rag.get_evidence("No space left on device")
    assert ev and {"source", "chunk_id", "content"} <= ev[0].keys()


def test_pipeline_no_context(kb):
    *_, retr = build(kb, max_distance=0.0, min_bm25=1e9)
    rag = RAGPipeline(retr, FakeListChatModel(responses=["x"]))
    assert rag.answer("weather in paris") == NO_CONTEXT_MESSAGE


def test_llm_retries_then_fails(kb):
    *_, retr = build(kb)
    calls = {"n": 0}

    def boom(_):
        calls["n"] += 1
        raise ConnectionError("rate limited")

    rag = RAGPipeline(retr, RunnableLambda(boom), max_retries=3, backoff=0)
    with pytest.raises(LLMGenerationError):
        rag.answer("No space left on device")
    assert calls["n"] == 3