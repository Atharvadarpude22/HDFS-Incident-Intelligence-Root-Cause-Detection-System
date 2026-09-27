import numpy as np
import pandas as pd


def to_jsonable(obj):
    """Recursively convert numpy / pandas values so json.dumps works."""
    if obj is pd.NaT or obj is None:
        return None
    if isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return None if (obj != obj or obj in (float("inf"), float("-inf"))) else obj
    if isinstance(obj, np.generic):
        return to_jsonable(obj.item())
    if isinstance(obj, np.ndarray):
        return [to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.Timedelta):
        return str(obj)
    if isinstance(obj, pd.DataFrame):
        return [to_jsonable(r) for r in obj.to_dict("records")]
    if isinstance(obj, pd.Series):
        return to_jsonable(obj.to_dict())
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    return str(obj)


class AgentTools:
    """The investigator's toolbox. Every tool returns JSON-safe data and raises on bad input."""

    TOOL_NAMES = (
        "get_incident", "get_logs", "get_features",
        "detect_anomaly", "search_runbook", "find_similar_incidents",
    )
    DESCRIPTIONS = {
        "get_incident": "Summary of the incident window (log counts, levels, signals).",
        "get_logs": "Log lines of the incident, most severe first.",
        "get_features": "Numeric features of the incident window.",
        "detect_anomaly": "ML anomaly score and verdict for the incident window.",
        "search_runbook": "Search the HDFS troubleshooting runbooks (RAG retriever).",
        "find_similar_incidents": "Other windows with a similar feature profile.",
    }
    SEVERITY_ORDER = {"FATAL": 0, "ERROR": 1, "WARN": 2, "UNKNOWN": 3, "DEBUG": 4, "INFO": 5}
    MAX_LOG_CHARS = 300
    MAX_RUNBOOK_CHARS = 600

    def __init__(self, incident_service, model_predictor, retriever):
        for name, dep in (("incident_service", incident_service),
                          ("model_predictor", model_predictor),
                          ("retriever", retriever)):
            if dep is None:
                raise ValueError(f"{name} is required.")
        self.incident_service = incident_service
        self.model_predictor = model_predictor
        self.retriever = retriever

    # ------------------------------------------------------------------
    def get_tool(self, name):
        if name not in self.TOOL_NAMES:
            raise ValueError(f"Unknown tool {name!r}. Available: {list(self.TOOL_NAMES)}")
        return getattr(self, name)

    def describe_tools(self):
        return dict(self.DESCRIPTIONS)

    # ------------------------------------------------------------------
    def get_incident(self, incident_id):
        return to_jsonable(self.incident_service.get_incident(incident_id))

    def get_logs(self, incident_id, limit=30):
        """Up to `limit` lines: ERROR/FATAL first, then WARN, then the rest."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer.")

        cid = self.incident_service.normalize_id(incident_id)
        df = self.incident_service.get_logs(cid)
        total = len(df)

        ranked = (
            df.assign(_rank=df["Level"].map(self.SEVERITY_ORDER).fillna(3))
            .sort_values(["_rank", "Timestamp"], kind="stable")
            .head(limit)
        )
        rows = [
            {
                "time": r["Timestamp"].isoformat(),
                "level": str(r["Level"]),
                "component": str(r.get("Component", "")),
                "event_id": str(r.get("EventId", "")),
                "content": str(r["Content"])[: self.MAX_LOG_CHARS],
            }
            for _, r in ranked.iterrows()
        ]
        return {"incident_id": cid, "total_logs": int(total), "shown": len(rows), "logs": rows}

    def get_features(self, incident_id):
        return to_jsonable(self.incident_service.get_features(incident_id))

    def detect_anomaly(self, incident_id):
        cid = self.incident_service.normalize_id(incident_id)
        features = self.incident_service.get_features(cid)
        prediction = self.model_predictor.predict([features])[0]
        return {
            "incident_id": cid,
            "anomaly_score": round(float(prediction["anomaly_score"]), 4),
            "is_anomaly": bool(prediction["is_anomaly"]),
        }

    def search_runbook(self, query, top_k=3):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")

        docs = self.retriever.retrieve(query, top_k=top_k) or []
        return {
            "query": query,
            "results": [self._normalize_doc(d, rank) for rank, d in enumerate(docs, start=1)],
        }

    def find_similar_incidents(self, incident_id, top_k=3):
        cid = self.incident_service.normalize_id(incident_id)
        return {
            "incident_id": cid,
            "similar": to_jsonable(self.incident_service.find_similar_incidents(cid, top_k=top_k)),
        }

    # ------------------------------------------------------------------
    def _normalize_doc(self, doc, rank):
        """Accepts LangChain Documents or dicts from the retriever."""
        if isinstance(doc, dict):
            content = doc.get("content") or doc.get("page_content") or ""
            meta = doc.get("metadata", doc)
        elif hasattr(doc, "page_content"):
            content, meta = doc.page_content, (getattr(doc, "metadata", None) or {})
        else:
            content, meta = str(doc), {}

        return to_jsonable({
            "rank": rank,
            "source": meta.get("source", "unknown"),
            "chunk_id": meta.get("chunk_id"),
            "dense_distance": meta.get("dense_distance"),
            "bm25_score": meta.get("bm25_score"),
            "content": str(content)[: self.MAX_RUNBOOK_CHARS],
        })