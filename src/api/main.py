"""Exposes the system through FastAPI.

Run with:  uvicorn src.api.main:app --reload
Docs at:   http://127.0.0.1:8000/docs
"""
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query

from src.services import IncidentNotFoundError

from .dependencies import Container, Settings, get_container
from .schemas import (
    AnalyzeRequest,
    ErrorResponse,
    HealthResponse,
    IncidentListItem,
    IncidentSummary,
    InvestigationRequest,
    InvestigationResponse,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up: loading data, model and RAG index ...")
    container = Container(Settings()).build()
    app.state.container = container
    logger.info("Startup complete: %s", container.health())
    yield


app = FastAPI(
    title="HDFS Incident Intelligence Agent",
    description=(
        "Give it an HDFS incident/log window, and it detects abnormal behaviour, "
        "searches HDFS troubleshooting knowledge, investigates the evidence, explains "
        "why it reached its conclusion, and returns a probable cause + recommended action."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


def _not_ready() -> HTTPException:
    return HTTPException(
        status_code=503,
        detail="Incident data is not available yet. Check GET /health for details, "
              "and run scripts/prepare_data.py first.",
    )


@app.get("/", tags=["system"])
def root():
    return {"name": "HDFS Incident Intelligence Agent", "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health(container: Container = Depends(get_container)):
    return container.health()


@app.post(
    "/investigate", response_model=InvestigationResponse, tags=["investigation"],
    responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def investigate(body: InvestigationRequest, container: Container = Depends(get_container)):
    """Full investigation of one incident window (get_logs, get_features, detect_anomaly,
    search_runbook, find_similar_incidents -> root cause + recommendations)."""
    if container.investigation_service is None:
        raise _not_ready()
    try:
        return container.investigation_service.investigate(body.incident_id)
    except IncidentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post(
    "/analyze", response_model=InvestigationResponse, tags=["investigation"],
    responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def analyze(body: AnalyzeRequest, container: Container = Depends(get_container)):
    """Free-text question. An incident id inside the text (e.g. 'INC-014') runs a full
    investigation; otherwise it's a runbook-only analysis of the description given."""
    if container.investigation_service is None:
        raise _not_ready()
    try:
        return container.investigation_service.analyze(body.query)
    except IncidentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get(
    "/incident/{incident_id}", response_model=IncidentSummary, tags=["incidents"],
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse},
              503: {"model": ErrorResponse}},
)
def get_incident(incident_id: str, container: Container = Depends(get_container)):
    """Summary of one incident window (log counts, levels, signals) - no ML/RAG/agent involved."""
    if container.incident_service is None:
        raise _not_ready()
    try:
        return container.incident_service.get_incident(incident_id)
    except IncidentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get(
    "/incidents", response_model=list[IncidentListItem], tags=["incidents"],
    responses={503: {"model": ErrorResponse}},
)
def list_incidents(
    limit: int = Query(20, ge=1, le=200),
    sort_by: str = Query("time", pattern="^(time|severity)$"),
    container: Container = Depends(get_container),
):
    """Browse incidents. sort_by='severity' is the quickest way to find something worth
    investigating; not in the original 3-endpoint outline, but small and useful for the UI."""
    if container.incident_service is None:
        raise _not_ready()
    return container.incident_service.list_incidents(limit=limit, sort_by=sort_by)