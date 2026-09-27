"""Defines what enters and leaves the API."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


def _not_blank(value):
    if not value or not value.strip():
        raise ValueError("must not be empty")
    return value.strip()


# ---------------------------------------------------------------- requests
class InvestigationRequest(BaseModel):
    incident_id: str = Field(..., examples=["INC-001"])

    @field_validator("incident_id")
    @classmethod
    def _clean(cls, v):
        return _not_blank(v)


class AnalyzeRequest(BaseModel):
    """Free-text version of /investigate; not in the original 3-endpoint outline,
    but reuses the same IncidentAgent, so it costs nothing extra to expose."""
    query: str = Field(..., examples=["DataNode heartbeat failure"])

    @field_validator("query")
    @classmethod
    def _clean(cls, v):
        return _not_blank(v)


# ---------------------------------------------------------------- responses
class InvestigationResponse(BaseModel):
    # --- fields named in the project outline ---
    incident_id: Optional[str] = None
    anomaly_score: Optional[float] = None
    root_cause: str
    confidence: float
    evidence: List[str]
    explanation: List[str]
    recommendations: List[str]

    # --- extra fields the agent also produces ---
    query: Optional[str] = None
    is_anomaly: Optional[bool] = None
    window_start: Optional[str] = None
    cause_scores: Dict[str, float] = Field(default_factory=dict)
    runbooks: List[str] = Field(default_factory=list)
    similar_incidents: List[Dict[str, Any]] = Field(default_factory=list)
    evidence_details: List[Dict[str, Any]] = Field(default_factory=list)
    ml_explanation: Optional[Dict[str, Any]] = None
    limitations: List[str] = Field(default_factory=list)
    narrative: Optional[str] = None
    llm_used: bool = False
    plan: List[Dict[str, Any]] = Field(default_factory=list)
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    generated_at: Optional[str] = None
    duration_ms: Optional[int] = None


class IncidentSummary(BaseModel):
    incident_id: str
    window_start: str
    window_end: str
    n_logs: int
    level_counts: Dict[str, int]
    n_errors: int
    n_warnings: int
    n_fatal: int
    components: List[str]
    signals: Dict[str, int]


class IncidentListItem(BaseModel):
    incident_id: str
    window_start: str
    n_logs: int
    fatal_count: int
    error_count: int
    warning_count: int


class HealthResponse(BaseModel):
    status: str                    # "ok" | "degraded"
    data_loaded: bool
    model_loaded: bool
    index_loaded: bool
    llm_enabled: bool
    n_incidents: Optional[int] = None
    details: Dict[str, str] = Field(default_factory=dict)   # component -> error message


class ErrorResponse(BaseModel):
    detail: str