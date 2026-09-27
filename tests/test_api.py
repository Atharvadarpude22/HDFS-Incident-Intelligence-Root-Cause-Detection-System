import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_agent as t  # reuses make_raw_logs / FakeRetriever from the agent tests

from src.data import DataPreprocessor
from src.ml import AnomalyDetector, ModelPredictor
from src.services import IncidentService
from src.agents import AgentTools, AgentPlanner, IncidentAgent
from src.services import InvestigationService
from src.explainability import IncidentExplainer
from src.api.dependencies import Container, Settings, get_container
from src.api.main import app

DATANODE_INCIDENT = t.DATANODE_INCIDENT   # "INC-026"
NORMAL_INCIDENT = t.NORMAL_INCIDENT       # "INC-004"


# ---------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def raw():
    return t.make_raw_logs()


@pytest.fixture(scope="module")
def container(raw, tmp_path_factory):
    model_dir = tmp_path_factory.mktemp("model")
    features = DataPreprocessor(window="1min").preprocess(raw)
    detector = AnomalyDetector(contamination=0.05).train(features)
    model_path = detector.save(model_dir / "m.pkl")

    service = IncidentService(t.FakeLoader(raw))
    predictor = ModelPredictor(model_path)
    predictor.load_model()

    tools = AgentTools(service, predictor, t.FakeRetriever())
    agent = IncidentAgent(AgentPlanner(), tools, explainer=IncidentExplainer(predictor.detector))

    c = Container(Settings())
    c.incident_service = service
    c.model_predictor = predictor
    c.retriever = t.FakeRetriever()
    c.explainer = agent.explainer
    c.investigation_service = InvestigationService(agent)
    return c


@pytest.fixture(scope="module")
def degraded_container():
    return Container(Settings())  # nothing built -> everything None


@pytest.fixture
def client(container):
    app.dependency_overrides[get_container] = lambda: container
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def degraded_client(degraded_container):
    app.dependency_overrides[get_container] = lambda: degraded_container
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ---------------------------------------------------------------- root & health
def test_root(client):
    r = client.get("/")
    assert r.status_code == 200 and r.json()["docs"] == "/docs"


def test_health_ok(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["data_loaded"] and body["model_loaded"]
    assert body["n_incidents"] == 40
    assert body["index_loaded"] is True                        # FakeRetriever isn't NullRetriever


def test_health_degraded(degraded_client):
    body = degraded_client.get("/health").json()
    assert body["status"] == "degraded" and body["data_loaded"] is False
    assert body["n_incidents"] is None


# ---------------------------------------------------------------- /investigate
def test_investigate_success(client):
    r = client.post("/investigate", json={"incident_id": DATANODE_INCIDENT})
    assert r.status_code == 200
    body = r.json()
    assert body["incident_id"] == DATANODE_INCIDENT
    assert body["root_cause"] == "DataNode failure"
    assert body["is_anomaly"] is True and 0 <= body["confidence"] <= 1
    assert body["evidence"] and body["recommendations"] and body["explanation"]
    assert body["ml_explanation"] is not None
    assert isinstance(body["steps"], list) and body["steps"]


def test_investigate_normal_window(client):
    body = client.post("/investigate", json={"incident_id": NORMAL_INCIDENT}).json()
    assert body["root_cause"] == "No abnormal behaviour detected"
    assert body["is_anomaly"] is False


def test_investigate_not_found(client):
    r = client.post("/investigate", json={"incident_id": "INC-999"})
    assert r.status_code == 404 and "INC-999" in r.json()["detail"]


def test_investigate_blank_id_returns_422(client):
    r = client.post("/investigate", json={"incident_id": "   "})
    assert r.status_code == 422                                 # pydantic validation, before our code runs


def test_investigate_missing_field_returns_422(client):
    assert client.post("/investigate", json={}).status_code == 422


def test_investigate_not_ready(degraded_client):
    r = degraded_client.post("/investigate", json={"incident_id": "INC-001"})
    assert r.status_code == 503


# ---------------------------------------------------------------- /analyze
def test_analyze_by_id(client):
    body = client.post("/analyze", json={"query": "please check inc-26 for me"}).json()
    assert body["incident_id"] == DATANODE_INCIDENT


def test_analyze_free_text(client):
    body = client.post("/analyze", json={"query": "DataNode heartbeat failure"}).json()
    assert body["root_cause"] == "DataNode failure" and body["incident_id"] is None


def test_analyze_blank_returns_422(client):
    assert client.post("/analyze", json={"query": ""}).status_code == 422


# ---------------------------------------------------------------- /incident/{id}
def test_get_incident_success(client):
    body = client.get(f"/incident/{DATANODE_INCIDENT}").json()
    assert body["incident_id"] == DATANODE_INCIDENT and body["n_errors"] == 25
    assert body["signals"]["heartbeat"] == 25


def test_get_incident_not_found(client):
    r = client.get("/incident/INC-999")
    assert r.status_code == 404


def test_get_incident_not_ready(degraded_client):
    assert degraded_client.get("/incident/INC-001").status_code == 503


# ---------------------------------------------------------------- /incidents
def test_list_incidents_default(client):
    body = client.get("/incidents").json()
    assert len(body) == 20 and body[0]["incident_id"] == "INC-001"      # default sort_by=time


def test_list_incidents_severity_and_limit(client):
    body = client.get("/incidents", params={"sort_by": "severity", "limit": 5}).json()
    assert len(body) == 5 and body[0]["incident_id"] == DATANODE_INCIDENT


def test_list_incidents_bad_sort_by_returns_422(client):
    assert client.get("/incidents", params={"sort_by": "bogus"}).status_code == 422


def test_list_incidents_bad_limit_returns_422(client):
    assert client.get("/incidents", params={"limit": 0}).status_code == 422
    assert client.get("/incidents", params={"limit": 1000}).status_code == 422


def test_list_incidents_not_ready(degraded_client):
    assert degraded_client.get("/incidents").status_code == 503


# ---------------------------------------------------------------- Settings / Container.build()
def test_settings_reads_env(monkeypatch, tmp_path):
    monkeypatch.setenv("HDFS_RAW_DIR", str(tmp_path / "raw"))
    monkeypatch.setenv("HDFS_WINDOW", "30s")
    monkeypatch.setenv("HDFS_USE_LLM", "true")
    s = Settings()
    assert s.raw_dir == tmp_path / "raw" and s.window == "30s" and s.use_llm is True


def test_container_build_end_to_end(tmp_path, monkeypatch):
    """Exercises the real Container.build() wiring (not the manually-assembled fixture
    above): real DataLoader, real trained model file, and a missing RAG index -
    all without any network access, since load_retriever now fails fast (see retriever.py)."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    t.make_raw_logs().to_csv(raw_dir / "HDFS_2k.log_structured.csv", index=False)

    features = DataPreprocessor(window="1min").preprocess(t.make_raw_logs())
    model_path = tmp_path / "m.pkl"
    AnomalyDetector(contamination=0.05).train(features).save(model_path)

    monkeypatch.setenv("HDFS_RAW_DIR", str(raw_dir))
    monkeypatch.setenv("HDFS_MODEL_PATH", str(model_path))
    monkeypatch.setenv("HDFS_INDEX_PATH", str(tmp_path / "no_such_index"))
    monkeypatch.setenv("HDFS_WINDOW", "1min")

    c = Container(Settings()).build()
    health = c.health()
    assert health["status"] == "ok" and health["data_loaded"] and health["model_loaded"]
    assert health["index_loaded"] is False and "index" in health["details"]
    assert c.investigation_service is not None

    report = c.investigation_service.investigate(DATANODE_INCIDENT)
    assert report["root_cause"] == "DataNode failure"
    assert any("search_runbook" in s.get("tool", "") for s in report["steps"])
    assert any(not s["ok"] for s in report["steps"] if s.get("tool") == "search_runbook")