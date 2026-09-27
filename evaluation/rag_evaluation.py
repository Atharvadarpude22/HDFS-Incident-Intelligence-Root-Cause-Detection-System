"""
RAG retrieval evaluation: does the retriever return the right troubleshooting
document for a query?

Metrics:
    Top-1 accuracy - the top-ranked chunk's source is the expected document.
    Top-k recall    - the expected document appears anywhere in the top k
                      chunks (k=3 by default, matching the project outline).
    MRR             - mean reciprocal rank of the expected document (bonus;
                      more informative than accuracy/recall alone since it
                      also rewards "close, but not first").

Usage:
    python evaluation/rag_evaluation.py
    python evaluation/rag_evaluation.py --eval-set my_queries.json
    python evaluation/rag_evaluation.py --out evaluation/rag_report.json

--eval-set expects a JSON file: a list of {"query": ..., "expected_source": ...}
objects, where expected_source matches a runbook filename as it appears in
data/knowledge_base/ (e.g. "hdfs_datanode_failure.md"). Without --eval-set, a
built-in default set covering the 5 planned runbooks is used - expand it (or
supply your own) as you add more troubleshooting documents.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # project root
sys.path.insert(0, str(ROOT))

from src.rag import load_retriever

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("rag_evaluation")

DEFAULT_EVAL_SET = [
    {"query": "DataNode is not sending heartbeats to the NameNode", "expected_source": "hdfs_datanode_failure.md"},
    {"query": "lost contact with a DataNode, is it dead", "expected_source": "hdfs_datanode_failure.md"},
    {"query": "DataNode process crashed with an out of memory error", "expected_source": "hdfs_datanode_failure.md"},
    {"query": "NameNode stuck in safe mode after restart", "expected_source": "hdfs_namenode_failure.md"},
    {"query": "fsimage or edit log failed to load on startup", "expected_source": "hdfs_namenode_failure.md"},
    {"query": "standby NameNode failover is not happening", "expected_source": "hdfs_namenode_failure.md"},
    {"query": "no space left on device on a DataNode volume", "expected_source": "hdfs_disk_failure.md"},
    {"query": "disk volume became read-only filesystem", "expected_source": "hdfs_disk_failure.md"},
    {"query": "checksum mismatch or corrupt block on disk", "expected_source": "hdfs_disk_failure.md"},
    {"query": "connection timed out between HDFS nodes", "expected_source": "hdfs_network_failure.md"},
    {"query": "socket exception broken pipe while writing a block", "expected_source": "hdfs_network_failure.md"},
    {"query": "clients cannot reach the NameNode, network unreachable", "expected_source": "hdfs_network_failure.md"},
    {"query": "blocks are under-replicated across the cluster", "expected_source": "hdfs_replication_failure.md"},
    {"query": "could only be replicated to 0 nodes instead of minReplication",
     "expected_source": "hdfs_replication_failure.md"},
    {"query": "fsck reports missing blocks", "expected_source": "hdfs_replication_failure.md"},
]


def load_eval_set(path=None):
    if path is None:
        return [dict(row) for row in DEFAULT_EVAL_SET]

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path} must contain a non-empty JSON list of "
                         "{query, expected_source} objects.")
    for i, row in enumerate(data):
        if not isinstance(row, dict) or "query" not in row or "expected_source" not in row:
            raise ValueError(f"Item {i} in {path} must have 'query' and 'expected_source'.")
    return data


def evaluate_retriever(retriever, eval_set, top_k=3):
    """Returns {n_queries, top_1_accuracy, top_k_recall, mrr, per_query: [...]}."""
    if not eval_set:
        raise ValueError("eval_set is empty.")
    if top_k < 1:
        raise ValueError("top_k must be >= 1.")

    rows = []
    hits_at_1 = hits_at_k = 0
    reciprocal_ranks = []

    for case in eval_set:
        query, expected = case["query"], case["expected_source"]
        try:
            results = retriever.retrieve_with_scores(query, top_k=top_k)
        except Exception as exc:  # a single bad query shouldn't abort the whole evaluation
            logger.warning("Query failed (%r): %s", query, exc)
            results = []

        sources = [r["source"] for r in results]
        rank = next((i + 1 for i, s in enumerate(sources) if s == expected), None)

        hit_1, hit_k = rank == 1, rank is not None
        hits_at_1 += int(hit_1)
        hits_at_k += int(hit_k)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

        rows.append({
            "query": query, "expected_source": expected, "retrieved_sources": sources,
            "rank": rank, "hit_at_1": hit_1, "hit_at_k": hit_k,
        })

    n = len(eval_set)
    return {
        "n_queries": n,
        "top_1_accuracy": hits_at_1 / n,
        "top_k_recall": hits_at_k / n,
        "top_k": top_k,
        "mrr": sum(reciprocal_ranks) / n,
        "per_query": rows,
    }


def run(index_path, eval_set_path=None, top_k=3, out_path=None, embedding_service=None):
    try:
        retriever = load_retriever(index_path, embedding_service=embedding_service,
                                   max_distance=float("inf"), min_bm25_score=0.0)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{exc} Build it first: python scripts/build_index.py") from exc

    eval_set = load_eval_set(eval_set_path)
    report = evaluate_retriever(retriever, eval_set, top_k=top_k)

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(report, indent=2))
        logger.info("Saved report to %s", out_path)
    return report


def _print_summary(report):
    print(f"Top-1 accuracy: {report['top_1_accuracy']:.0%}")
    print(f"Top-{report['top_k']} recall:  {report['top_k_recall']:.0%}")
    print(f"MRR:            {report['mrr']:.3f}")
    for row in report["per_query"]:
        mark = "OK  " if row["hit_at_1"] else ("~   " if row["hit_at_k"] else "MISS")
        print(f"  [{mark}] {row['query']!r} -> expected {row['expected_source']}, "
             f"got {row['retrieved_sources']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--index", default=str(ROOT / "vector_db"))
    p.add_argument("--eval-set", default=None,
                  help="optional JSON file of [{query, expected_source}, ...]")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--out", default=None, help="optional path to save the report as JSON")
    args = p.parse_args()

    result = run(args.index, args.eval_set, args.top_k, args.out)
    _print_summary(result)