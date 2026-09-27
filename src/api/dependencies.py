"""
Builds every heavy component (data, model, RAG index, agent) once and holds
it for the app's lifetime. Not named in the outline's src/api/ file list, but
kept separate from main.py so it can be unit-tested and swapped in tests
without hitting real files, models or the network (see FastAPI's recommended
dependency-injection pattern).
"""
import logging
import os
from pathlib import Path

from fastapi import Request

from src.agents import AgentPlanner, AgentTools, IncidentAgent
from src.data import DataLoader, DataPreprocessor
from src.explainability import IncidentExplainer
from src.ml import ModelPredictor
from src.services import IncidentService, InvestigationService

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]  # project root


class Settings:
    """Paths and toggles, all overridable via environment variables."""

    def __init__(self):
        self.raw_dir = Path(os.getenv("HDFS_RAW_DIR", str(ROOT / "data" / "raw" / "hdfs")))
        self.model_path = Path(os.getenv("HDFS_MODEL_PATH", str(ROOT / "models" / "anomaly_model.pkl")))
        self.index_path = Path(os.getenv("HDFS_INDEX_PATH", str(ROOT / "vector_db")))
        self.window = os.getenv("HDFS_WINDOW", "1min")
        self.use_llm = os.getenv("HDFS_USE_LLM", "false").strip().lower() == "true"


class NullRetriever:
    """Stands in for the retriever when the vector index isn't built yet.
    search_runbook then fails clearly (caught by IncidentAgent) instead of crashing startup."""

    def retrieve(self, query, top_k=3):
        raise RuntimeError("RAG index is not available. Run scripts/build_index.py first.")


class Container:
    """
    Wires: DataLoader -> IncidentService
           ModelPredictor (+ IncidentExplainer)
           Retriever (RAG)
           -> AgentTools -> IncidentAgent -> InvestigationService

    Any single component that fails to load is recorded in `errors` and
    replaced with a safe stand-in, so the API still starts and /health
    reports exactly what's missing rather than crashing on import.
    """

    def __init__(self, settings=None):
        self.settings = settings or Settings()
        self.errors = {}
        self.incident_service = None
        self.model_predictor = None
        self.retriever = None
        self.explainer = None
        self.investigation_service = None

    def build(self):
        s = self.settings

        try:
            self.incident_service = IncidentService(
                DataLoader(s.raw_dir), preprocessor=DataPreprocessor(window=s.window))
            self.incident_service.load()
        except Exception as exc:
            logger.error("Could not load incident data from %s: %s", s.raw_dir, exc)
            self.errors["data"] = str(exc)

        self.model_predictor = ModelPredictor(s.model_path)
        try:
            self.model_predictor.load_model()
        except Exception as exc:
            logger.warning("Anomaly model not available at %s: %s", s.model_path, exc)
            self.errors["model"] = str(exc)  # detect_anomaly will retry & fail per-call

        try:
            from src.rag import load_retriever
            self.retriever = load_retriever(s.index_path)
        except Exception as exc:
            logger.warning("RAG index not available at %s: %s", s.index_path, exc)
            self.errors["index"] = str(exc)
            self.retriever = NullRetriever()

        if self.model_predictor.detector is not None:
            self.explainer = IncidentExplainer(self.model_predictor.detector)

        llm = None
        if s.use_llm:
            try:
                from src.rag import build_llm
                llm = build_llm()
            except Exception as exc:
                logger.warning("LLM not available: %s", exc)
                self.errors["llm"] = str(exc)

        if self.incident_service is not None:
            tools = AgentTools(self.incident_service, self.model_predictor, self.retriever)
            agent = IncidentAgent(AgentPlanner(), tools, llm=llm, explainer=self.explainer)
            self.investigation_service = InvestigationService(agent)

        return self

    def health(self):
        n_incidents = None
        if self.incident_service is not None:
            try:
                n_incidents = self.incident_service.n_incidents
            except Exception:
                n_incidents = None
        return {
            "status": "ok" if self.investigation_service is not None else "degraded",
            "data_loaded": self.incident_service is not None,
            "model_loaded": bool(self.model_predictor and self.model_predictor.detector),
            "index_loaded": self.retriever is not None and not isinstance(self.retriever, NullRetriever),
            "llm_enabled": self.settings.use_llm,
            "n_incidents": n_incidents,
            "details": self.errors,
        }


def get_container(request: Request) -> Container:
    """FastAPI dependency. The real Container is built once in main.lifespan()
    and stored on app.state; tests override this function directly."""
    return request.app.state.container