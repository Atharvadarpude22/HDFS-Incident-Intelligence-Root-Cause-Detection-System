import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import DataPreprocessor
from src.ml import AnomalyDetector, ModelPredictor
from src.agents import AgentTools, AgentPlanner, IncidentAgent, to_jsonable
from src.services import IncidentService, IncidentNotFoundError, InvestigationService
from src.explainability import IncidentExplainer

DATANODE_INCIDENT = "INC-026"   # minute 25: heartbeat errors
NETWORK_INCIDENT = "INC-011"    # minute 10: connection reset warnings
NORMAL_INCIDENT = "INC-004"


# ---------------------------------------------------------------- fixtures
def _row(lid, ts, level, content, pid=100):
    return dict(LineId=lid, Date=ts.strftime("%y%m%d"), Time=ts.strftime("%H%M%S"), Pid=pid,
                Level=level, Component="dfs.DataNode", Content=content, EventId="E1",
                EventTemplate="t")


def make_raw_logs():
    rows, lid = [], 1
    start = pd.Timestamp("2008-11-09 20:00:00")
    for minute in range(40):
        for sec in range(0, 60, 6):
            ts = start + pd.Timedelta(minutes=minute, seconds=sec)
            rows.append(_row(lid, ts, "INFO", f"PacketResponder 1 for block blk_{lid} terminating", 100 + sec))
            lid += 1
    for k in range(25):                                   # DataNode incident
        ts = start + pd.Timedelta(minutes=25, seconds=k * 2)
        rows.append(_row(lid, ts, "ERROR", f"Lost heartbeat from datanode 10.0.0.{k} block blk_{lid}", 300 + k))
        lid += 1
    for k in range(8):                                    # Network incident (WARN only)
        ts = start + pd.Timedelta(minutes=10, seconds=k * 5)
        rows.append(_row(lid, ts, "WARN", f"Connection reset by peer 10.0.0.{k}", 200 + k))
        lid += 1
    return pd.DataFrame(rows)


class FakeLoader:
    def __init__(self, df):
        self.df = df

    def load_structured_logs(self):
        return self.df.copy()


class FakeRetriever:
    def __init__(self):
        self.queries = []

    def retrieve(self, query, top_k=3):
        self.queries.append(query)
        return [
            SimpleNamespace(page_content="DataNode heartbeat failure: check node status.",
                            metadata={"source": "hdfs_datanode_failure.md", "chunk_id": 1,
                                      "dense_distance": 0.4, "bm25_score": 3.0}),
            {"content": "Network timeouts: check firewall.", "source": "hdfs_network_failure.md",
             "chunk_id": 2},
        ][:top_k]


class BrokenRetriever:
    def retrieve(self, query, top_k=3):
        raise ConnectionError("vector db down")


@pytest.fixture(scope="module")
def raw():
    return make_raw_logs()


@pytest.fixture(scope="module")
def model_path(raw, tmp_path_factory):
    features = DataPreprocessor(window="1min").preprocess(raw)
    detector = AnomalyDetector(contamination=0.05).train(features)
    return detector.save(tmp_path_factory.mktemp("model") / "m.pkl")


@pytest.fixture
def service(raw):
    return IncidentService(FakeLoader(raw))


@pytest.fixture
def detector(raw):
    features = DataPreprocessor(window="1min").preprocess(raw)
    return AnomalyDetector(contamination=0.05).train(features)


@pytest.fixture
def make_agent(service, model_path):
    def build(retriever=None, predictor=None, llm=None, explainer=None):
        tools = AgentTools(service, predictor or ModelPredictor(model_path),
                           retriever or FakeRetriever())
        return IncidentAgent(AgentPlanner(), tools, llm, explainer=explainer)
    return build


# ---------------------------------------------------------------- IncidentService
def test_service_ids_and_summary(service):
    assert service.n_incidents == 40
    assert service.normalize_id("inc-26") == DATANODE_INCIDENT
    assert service.normalize_id(26) == DATANODE_INCIDENT
    inc = service.get_incident(DATANODE_INCIDENT)
    assert inc["n_errors"] == 25 and inc["signals"]["heartbeat"] == 25
    assert service.get_incident(NETWORK_INCIDENT)["n_warnings"] == 8
    assert service.list_incidents(sort_by="severity")[0]["incident_id"] == DATANODE_INCIDENT
    assert len(service.list_incidents(limit=5)) == 5


def test_service_logs_features_similar(service):
    logs = service.get_logs(DATANODE_INCIDENT, limit=5)
    assert len(logs) == 5 and logs["Timestamp"].is_monotonic_increasing
    feats = service.get_features(DATANODE_INCIDENT)
    json.dumps(feats)
    assert feats["heartbeat_failures"] == 25
    sim = service.find_similar_incidents(NORMAL_INCIDENT, top_k=3)
    assert len(sim) == 3 and all(s["incident_id"] != NORMAL_INCIDENT for s in sim)


@pytest.mark.parametrize("bad", ["INC-999", "abc", "INC-000", 0, -1])
def test_service_not_found(service, bad):
    with pytest.raises(IncidentNotFoundError):
        service.get_incident(bad)


def test_service_bad_input(service):
    with pytest.raises(ValueError):
        service.get_incident("  ")
    with pytest.raises(TypeError):
        service.get_incident(None)
    with pytest.raises(ValueError):
        service.get_logs("INC-001", limit=0)
    with pytest.raises(ValueError):
        IncidentService(FakeLoader(pd.DataFrame({"x": [1]}))).load()   # blocking validation error


# ---------------------------------------------------------------- AgentTools
def test_tools_outputs(service, model_path):
    tools = AgentTools(service, ModelPredictor(model_path), FakeRetriever())
    logs = tools.get_logs(DATANODE_INCIDENT, limit=10)
    assert logs["shown"] == 10 and logs["total_logs"] == 35
    assert all(l["level"] == "ERROR" for l in logs["logs"])                # severe first

    det = tools.detect_anomaly(DATANODE_INCIDENT)
    assert det["is_anomaly"] and 0.5 < det["anomaly_score"] <= 1
    assert not tools.detect_anomaly(NORMAL_INCIDENT)["is_anomaly"]

    res = tools.search_runbook("datanode heartbeat")["results"]
    assert [r["source"] for r in res] == ["hdfs_datanode_failure.md", "hdfs_network_failure.md"]
    json.dumps(tools.get_incident("INC-001"))


def test_tools_validation(service, model_path):
    tools = AgentTools(service, ModelPredictor(model_path), FakeRetriever())
    for bad_query in ["", "  ", None]:
        with pytest.raises(ValueError):
            tools.search_runbook(bad_query)
    with pytest.raises(ValueError):
        tools.get_tool("delete_everything")
    with pytest.raises(ValueError):
        AgentTools(service, None, FakeRetriever())
    assert set(tools.describe_tools()) == set(AgentTools.TOOL_NAMES)


def test_to_jsonable():
    import numpy as np
    out = to_jsonable({"a": np.float32(1.5), "b": np.int64(2), "c": pd.Timestamp("2008-11-09"),
                       "d": float("nan"), "e": np.array([1, 2]), "f": pd.NaT})
    json.dumps(out)
    assert out["d"] is None and out["f"] is None


# ---------------------------------------------------------------- AgentPlanner
def test_planner_adapts_to_incident(service):
    planner = AgentPlanner()
    plan = planner.create_plan(service.get_incident(DATANODE_INCIDENT))
    queries = [s["args"]["query"] for s in plan if s["tool"] == "search_runbook"]
    assert "HDFS DataNode heartbeat failure" in queries and "$auto" in queries
    assert plan[-1]["tool"] == "find_similar_incidents"

    clean = planner.create_plan(service.get_incident(NORMAL_INCIDENT))
    assert [s["args"]["query"] for s in clean if s["tool"] == "search_runbook"] == ["$auto"]
    assert planner.select_tools(plan)[:3] == ["get_logs", "get_features", "detect_anomaly"]


def test_planner_errors():
    planner = AgentPlanner()
    with pytest.raises(ValueError):
        planner.create_plan({})
    with pytest.raises(ValueError):
        planner.select_tools([{"tool": "rm -rf"}])
    with pytest.raises(ValueError):
        planner.select_tools([])
    with pytest.raises(ValueError):
        AgentPlanner(log_limit=0)


# ---------------------------------------------------------------- IncidentAgent
def test_investigate_datanode_incident(make_agent):
    report = make_agent().investigate(DATANODE_INCIDENT)
    json.dumps(report)
    assert report["root_cause"] == "DataNode failure"
    assert report["is_anomaly"] and report["confidence"] > 0.5
    assert report["evidence"] and report["recommendations"] and report["explanation"]
    assert "hdfs_datanode_failure.md" in report["runbooks"]
    assert report["similar_incidents"] and all(s["ok"] for s in report["steps"])
    assert report["narrative"] is None and report["llm_used"] is False


def test_root_cause_comes_from_logs_not_just_rag(make_agent):
    report = make_agent().investigate(NETWORK_INCIDENT)     # retriever still ranks DataNode first
    assert report["root_cause"] == "Network failure"


def test_normal_incident(make_agent):
    report = make_agent().investigate(NORMAL_INCIDENT)
    assert report["root_cause"] == "No abnormal behaviour detected"
    assert report["is_anomaly"] is False


def test_unknown_incident_propagates(make_agent):
    with pytest.raises(IncidentNotFoundError):
        make_agent().investigate("INC-999")


def test_graceful_degradation_retriever_down(make_agent):
    report = make_agent(retriever=BrokenRetriever()).investigate(DATANODE_INCIDENT)
    assert report["root_cause"] == "DataNode failure"               # still decided from logs + features
    assert any("search_runbook" in l for l in report["limitations"])
    assert any(not s["ok"] for s in report["steps"])


def test_graceful_degradation_model_missing(make_agent, tmp_path):
    report = make_agent(predictor=ModelPredictor(tmp_path / "missing.pkl")).investigate(DATANODE_INCIDENT)
    assert report["anomaly_score"] is None
    assert report["root_cause"] == "DataNode failure"
    assert any("anomaly detection was unavailable" in l for l in report["limitations"])


def test_llm_narrative_and_failure(make_agent):
    ok = make_agent(llm=lambda prompt: "The DataNode lost heartbeats.").investigate(DATANODE_INCIDENT)
    assert ok["llm_used"] and "DataNode" in ok["narrative"]

    def broken(prompt):
        raise TimeoutError("llm timeout")
    bad = make_agent(llm=broken).investigate(DATANODE_INCIDENT)
    assert bad["narrative"] is None and not bad["llm_used"]
    assert any("LLM narrative unavailable" in l for l in bad["limitations"])

    class ChatModel:                                       # LangChain-style object
        def invoke(self, prompt):
            return SimpleNamespace(content="chat answer")
    assert make_agent(llm=ChatModel()).investigate(DATANODE_INCIDENT)["narrative"] == "chat answer"


def test_auto_query_uses_evidence(service, model_path):
    retriever = FakeRetriever()
    IncidentAgent(AgentPlanner(), AgentTools(service, ModelPredictor(model_path), retriever)
                  ).investigate(DATANODE_INCIDENT)
    auto = retriever.queries[-1]
    assert auto.startswith("HDFS") and "heartbeat" in auto.lower() and "blk_<id>" in auto


def test_analyze(make_agent):
    agent = make_agent()
    by_id = agent.analyze("please look at inc-26 for me")
    assert by_id["incident_id"] == DATANODE_INCIDENT

    free = agent.analyze("DataNode heartbeat failure")
    assert free["root_cause"] == "DataNode failure" and free["incident_id"] is None

    vague = agent.analyze("something feels slow")
    assert vague["root_cause"] in ("Anomalous behaviour, cause undetermined",)

    for bad in ["", "   ", None]:
        with pytest.raises(ValueError):
            agent.analyze(bad)


def test_execute_tool(make_agent):
    agent = make_agent()
    with pytest.raises(ValueError):
        agent.execute_tool("format_disk", {})
    failed = agent.execute_tool("detect_anomaly", {"wrong_arg": 1})
    assert failed["ok"] is False and "TypeError" in failed["error"]
    assert agent.execute_tool("get_features", {"incident_id": "INC-001"})["ok"]


# ---------------------------------------------------------------- InvestigationService
def test_investigation_service(make_agent):
    agent = make_agent()
    calls = {"n": 0}
    original = agent.investigate
    agent.investigate = lambda i: (calls.__setitem__("n", calls["n"] + 1), original(i))[1]

    svc = InvestigationService(agent)
    first = svc.investigate("INC-026")
    again = svc.investigate("inc-26")                      # same incident, cached
    assert first is again and calls["n"] == 1
    svc.investigate("INC-026", refresh=True)
    assert calls["n"] == 2

    svc.analyze("network timeout")
    history = svc.get_history(limit=2)
    assert len(history) == 2 and history[0]["query"] == "network timeout"

    with pytest.raises(IncidentNotFoundError):
        svc.investigate("INC-999")
    with pytest.raises(ValueError):
        svc.get_history(0)
    with pytest.raises(ValueError):
        InvestigationService(None)


# ---------------------------------------------------------------- IncidentExplainer integration
def test_report_includes_ml_explanation(make_agent, detector):
    report = make_agent(explainer=IncidentExplainer(detector)).investigate(DATANODE_INCIDENT)
    assert report["ml_explanation"] is not None
    assert report["ml_explanation"]["top_factors"]
    assert any("ML factor:" in e for e in report["evidence"])
    json.dumps(report)


def test_report_without_explainer_has_no_ml_explanation(make_agent):
    report = make_agent().investigate(DATANODE_INCIDENT)
    assert report["ml_explanation"] is None
    assert not any("ML factor:" in e for e in report["evidence"])


def test_explainer_failure_does_not_break_report(make_agent, detector, monkeypatch):
    explainer = IncidentExplainer(detector)
    monkeypatch.setattr(explainer, "explain", lambda features: (_ for _ in ()).throw(RuntimeError("boom")))
    report = make_agent(explainer=explainer).investigate(DATANODE_INCIDENT)
    assert report["ml_explanation"] is None
    assert report["root_cause"] == "DataNode failure"           # investigation still completes


def test_explainer_independent_of_predictor(make_agent, detector, tmp_path):
    # The explainer wraps its own trained detector, so it can still explain a window
    # even if the *deployed* model file used by detect_anomaly is missing/broken.
    report = make_agent(predictor=ModelPredictor(tmp_path / "missing.pkl"),
                        explainer=IncidentExplainer(detector)).investigate(DATANODE_INCIDENT)
    assert report["anomaly_score"] is None                      # detect_anomaly tool failed
    assert report["ml_explanation"] is not None                 # explainer used its own model
    assert report["ml_explanation"]["top_factors"]
    # factor lines are only added alongside a verdict (detect_anomaly), which failed here
    assert not any("ML factor:" in e for e in report["evidence"])