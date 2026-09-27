"""
HDFS Incident Intelligence Agent - Streamlit UI.

Talks only to the FastAPI backend (src/api). Start the backend first:

    python -m uvicorn src.api.main:app --reload

Then run this app:

    streamlit run app/streamlit_app.py

Configure a different backend URL with the HDFS_API_URL environment
variable, or from the sidebar once the app is running.
"""
import os

import pandas as pd
import requests
import streamlit as st

DEFAULT_API_URL = os.getenv("HDFS_API_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT = 120  # seconds; an LLM-backed /investigate call can be slow
EM_DASH = "\u2014"  # kept as a plain variable, not inline in an f-string: a
                    # backslash escape inside an f-string's {expression} part
                    # is a SyntaxError on Python <3.12 (PEP 701 lifted this in 3.12)

st.set_page_config(page_title="HDFS Incident Intelligence", page_icon="\U0001F6A8", layout="wide")


# ---------------------------------------------------------------------------
# API client - every call to the backend goes through here
# ---------------------------------------------------------------------------
class APIError(Exception):
    """Carries a message already safe to show the user, plus the HTTP status if any."""

    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


def _extract_detail(response):
    try:
        body = response.json()
    except ValueError:
        return response.text[:300] or f"HTTP {response.status_code}"

    detail = body.get("detail")
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list):  # FastAPI/Pydantic validation errors (422)
        return "; ".join(
            f"{'.'.join(str(p) for p in e.get('loc', []) if p != 'body')}: {e.get('msg', e)}"
            for e in detail
        )
    return str(detail) if detail else f"HTTP {response.status_code}"


def call_api(method, path, base_url, **kwargs):
    """GET/POST to the backend. Raises APIError with a message ready to display."""
    url = base_url.rstrip("/") + path
    try:
        response = requests.request(method, url, timeout=REQUEST_TIMEOUT, **kwargs)
    except requests.exceptions.ConnectTimeout:
        raise APIError(f"Timed out connecting to {url}.")
    except requests.exceptions.ConnectionError:
        raise APIError(
            f"Can't reach the API at {base_url}. Is it running? "
            f"Start it with: python -m uvicorn src.api.main:app --reload"
        )
    except requests.exceptions.Timeout:
        raise APIError("The API took too long to respond (investigation may still be heavy on first run).")
    except requests.exceptions.RequestException as exc:
        raise APIError(f"Request failed: {exc}")

    if response.status_code >= 400:
        raise APIError(_extract_detail(response), status_code=response.status_code)
    try:
        return response.json()
    except ValueError:
        raise APIError("The API returned a response that wasn't valid JSON.")


def get_health(base_url):
    return call_api("GET", "/health", base_url)


def list_incidents(base_url, limit=20, sort_by="severity"):
    return call_api("GET", "/incidents", base_url, params={"limit": limit, "sort_by": sort_by})


def get_incident(base_url, incident_id):
    return call_api("GET", f"/incident/{incident_id}", base_url)


def investigate(base_url, incident_id):
    return call_api("POST", "/investigate", base_url, json={"incident_id": incident_id})


def analyze(base_url, query):
    return call_api("POST", "/analyze", base_url, json={"query": query})


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def render_health_badge(base_url):
    try:
        health = get_health(base_url)
    except APIError as exc:
        st.sidebar.error(f"\u274C {exc}")
        return None

    if health["status"] == "ok":
        st.sidebar.success("\u2705 API is up")
    else:
        st.sidebar.warning("\u26A0\uFE0F API is degraded")

    st.sidebar.caption(
        f"Incidents loaded: {health['n_incidents'] if health['n_incidents'] is not None else EM_DASH}"
    )
    cols = st.sidebar.columns(3)
    cols[0].metric("Data", "OK" if health["data_loaded"] else EM_DASH)
    cols[1].metric("Model", "OK" if health["model_loaded"] else EM_DASH)
    cols[2].metric("RAG", "OK" if health["index_loaded"] else EM_DASH)

    if health["details"]:
        with st.sidebar.expander("Component errors"):
            for component, message in health["details"].items():
                st.write(f"**{component}**: {message}")
    return health


def render_report(report):
    """Renders an InvestigationResponse (see src/api/schemas.py) - same shape whether
    it came from /investigate (incident_id set) or /analyze (query set)."""
    st.divider()
    header = report.get("incident_id") or f"Query: \u201C{report.get('query')}\u201D"
    st.subheader(header)

    score = report.get("anomaly_score")
    confidence = report.get("confidence") or 0.0

    col1, col2, col3 = st.columns(3)
    with col1:
        if score is not None:
            st.metric(
                "Anomaly Score", f"{score:.2f}",
                delta="ABNORMAL" if report.get("is_anomaly") else "normal",
                delta_color="inverse" if report.get("is_anomaly") else "normal",
            )
        else:
            st.metric("Anomaly Score", "N/A")
    with col2:
        st.metric("Likely Root Cause", report["root_cause"])
    with col3:
        st.metric("Confidence", f"{confidence:.0%}")
    st.progress(min(max(confidence, 0.0), 1.0))

    left, right = st.columns(2)
    with left:
        st.markdown("**Evidence**")
        for item in report.get("evidence", []):
            st.markdown(f"- {item}")
        if report.get("runbooks"):
            st.markdown("**Matched runbooks**")
            for source in report["runbooks"]:
                st.markdown(f"- `{source}`")
        if report.get("similar_incidents"):
            st.markdown("**Similar incidents**")
            st.dataframe(pd.DataFrame(report["similar_incidents"]), hide_index=True, width='stretch')

    with right:
        st.markdown("**Recommended Actions**")
        for i, action in enumerate(report.get("recommendations", []), start=1):
            st.markdown(f"{i}. {action}")
        st.markdown("**Why**")
        for line in report.get("explanation", []):
            st.markdown(f"- {line}")

    if report.get("narrative"):
        st.markdown("**AI narrative**")
        st.info(report["narrative"])

    ml_explanation = report.get("ml_explanation") or {}
    if ml_explanation.get("top_factors"):
        with st.expander("ML factors behind the anomaly score"):
            st.dataframe(pd.DataFrame(ml_explanation["top_factors"]), hide_index=True, width='stretch')

    if report.get("limitations"):
        st.warning("**Limitations of this report**\n\n" + "\n".join(f"- {l}" for l in report["limitations"]))

    with st.expander("Investigation steps (what the agent actually did)"):
        for step in report.get("steps", []):
            icon = "\u2705" if step.get("ok") else "\u274C"
            line = f"{icon} **{step['tool']}** \u2014 {step.get('purpose', '')}"
            if not step.get("ok"):
                line += f"  \n  \u21B3 error: {step.get('error')}"
            st.markdown(line)

    with st.expander("Raw JSON"):
        st.json(report)


def run_investigation(base_url, incident_id):
    with st.spinner(f"Investigating {incident_id} ..."):
        try:
            st.session_state.report = investigate(base_url, incident_id)
        except APIError as exc:
            st.session_state.report = None
            st.error(str(exc))


def run_analysis(base_url, query):
    with st.spinner("Analyzing ..."):
        try:
            st.session_state.report = analyze(base_url, query)
        except APIError as exc:
            st.session_state.report = None
            st.error(str(exc))


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
st.title("\U0001F6A8 HDFS Incident Intelligence")
st.caption(
    "Detects abnormal HDFS behaviour, searches troubleshooting knowledge, "
    "and explains its probable root cause with recommended actions."
)

with st.sidebar:
    st.header("Connection")
    base_url = st.text_input("API URL", value=st.session_state.get("base_url", DEFAULT_API_URL))
    st.session_state.base_url = base_url
    if st.button("\U0001F504 Refresh status", width='stretch'):
        st.rerun()
    health = render_health_badge(base_url)

if "report" not in st.session_state:
    st.session_state.report = None

investigate_tab, browse_tab, describe_tab = st.tabs(
    ["Investigate by ID", "Browse incidents", "Describe the problem"]
)

with investigate_tab:
    st.markdown("Enter an incident window id (e.g. `INC-001`) and investigate it.")
    col1, col2 = st.columns([3, 1])
    incident_id = col1.text_input("Incident ID", placeholder="INC-001", label_visibility="collapsed")
    if col2.button("INVESTIGATE", type="primary", width='stretch'):
        if not incident_id.strip():
            st.warning("Enter an incident ID first.")
        else:
            run_investigation(base_url, incident_id.strip())

with browse_tab:
    st.markdown("Incidents ranked by severity - pick one to investigate.")
    c1, c2 = st.columns(2)
    sort_by = c1.selectbox("Sort by", ["severity", "time"], index=0)
    limit = c2.slider("Show", min_value=5, max_value=100, value=20, step=5)

    if health is None:
        st.info("Connect to the API to browse incidents.")
    else:
        try:
            rows = list_incidents(base_url, limit=limit, sort_by=sort_by)
        except APIError as exc:
            rows = []
            st.error(str(exc))

        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, hide_index=True, width='stretch')
            picked = st.selectbox("Investigate one of these", [r["incident_id"] for r in rows])
            if st.button("INVESTIGATE SELECTED", type="primary"):
                run_investigation(base_url, picked)
        else:
            st.info("No incidents found yet. Run scripts/prepare_data.py first.")

with describe_tab:
    st.markdown("No incident id handy? Describe what you're seeing instead.")
    query = st.text_area(
        "What's happening?",
        placeholder="e.g. DataNode keeps losing heartbeats and replication warnings are showing up",
    )
    if st.button("ANALYZE", type="primary"):
        if not query.strip():
            st.warning("Describe the problem first.")
        else:
            run_analysis(base_url, query.strip())

if st.session_state.report:
    render_report(st.session_state.report)