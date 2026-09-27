# HDFS Incident Intelligence Agent

Give it an HDFS incident (a window of logs) and it detects abnormal
behavior, searches HDFS troubleshooting knowledge, investigates the
evidence, explains why it reached its conclusion, and returns a probable
root cause with recommended actions - the way an on-call engineer would
investigate, automated.

Built as a single vertical pipeline covering data engineering, unsupervised
ML, RAG, a rule-based agent, SHAP explainability, and a FastAPI + Streamlit
service layer. See [`docs/architecture.md`](docs/architecture.md) for the
full design writeup and [`docs/evaluation.md`](docs/evaluation.md) for
real, honestly-reported evaluation results (not just methodology).

## What it does

```
HDFS logs → features → anomaly score → agent investigation → explained root cause → API → UI
```

Given an incident window, the system combines four kinds of evidence before
deciding anything:

- **ML** - an Isolation Forest anomaly score (unsupervised - no labeled
  failure data exists for HDFS_2k, so it learns "normal" from the data
  itself), explained feature-by-feature with SHAP.
- **Logs** - pattern matches against known HDFS failure signatures
  (DataNode, NameNode, disk, network, replication).
- **RAG** - hybrid dense + sparse retrieval over troubleshooting runbooks.
- **History** - similar past incidents, by feature-space distance.

**Root cause is decided by a rule-based evidence scorer, not by an LLM** - a
cause is only named if it has direct evidence (log content or ML features),
not just a runbook match. An LLM, if configured, only narrates a decision
that's already been made deterministically. This makes the output
reproducible, testable, and correct even with no LLM available. See
[`docs/architecture.md`](docs/architecture.md#src-agents---the-part-that-makes-this-agentic)
for the full reasoning.

## Quickstart

```bash
git clone <this-repo>
cd it-incident-intelligence-agent
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env   # optional - defaults work out of the box
```

Put `HDFS_2k.log_structured.csv` (and optionally the raw `.log` and
`.log_templates.csv` files) in `data/raw/hdfs/`, then run the pipeline in
order:

```bash
python scripts/prepare_data.py           # logs -> data/processed/{logs,features}.csv
python scripts/train_model.py            # features -> models/anomaly_model.pkl
python scripts/build_index.py            # data/knowledge_base -> vector_db/ (needs internet, once)
```

Then run the API and/or the UI:

```bash
python -m uvicorn src.api.main:app --reload    # http://127.0.0.1:8000/docs
streamlit run app/streamlit_app.py             # talks to the API above
```

Check `GET /health` (or the Streamlit sidebar) to see what's loaded - the
system starts even if a step above was skipped, and reports exactly what's
missing rather than failing outright.

## Try it

```bash
curl -X POST localhost:8000/investigate \
  -H "Content-Type: application/json" \
  -d '{"incident_id": "INC-001"}'

curl -X POST localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "DataNode keeps losing heartbeats"}'

curl "localhost:8000/incidents?sort_by=severity&limit=5"
```

Incidents are numbered chronologically (`INC-001`, `INC-002`, ...) - each
one is a time window of logs, since HDFS_2k has no native incident IDs. Use
`GET /incidents?sort_by=severity` to find one worth investigating.

## Project layout

```
src/
├── data/            DataLoader, DataValidator, DataPreprocessor
├── ml/               AnomalyDetector (Isolation Forest), ModelTrainer, ModelPredictor
├── rag/               DocumentLoader/Chunker, EmbeddingService, VectorStore, hybrid Retriever
├── agents/            AgentTools, AgentPlanner, IncidentAgent (the decision-maker)
├── services/          IncidentService, InvestigationService
├── explainability/    IncidentExplainer (SHAP)
└── api/               FastAPI: schemas.py, dependencies.py, main.py

scripts/       prepare_data.py, train_model.py, build_index.py
evaluation/    model_evaluation.py, rag_evaluation.py
app/           streamlit_app.py (talks only to the API, over HTTP)
tests/         129 tests across every layer above
docs/          architecture.md, evaluation.md
```

Full pipeline diagram and per-module design notes:
[`docs/architecture.md`](docs/architecture.md).

## Testing

```bash
python -m pytest tests -v
```

129 tests across data, ML, RAG, agents, services, explainability, and the
API - all offline (a deterministic fake embedding model stands in for the
real HuggingFace model, so the suite never needs network access).

## Evaluation

```bash
python evaluation/model_evaluation.py     # anomaly detector: unsupervised + synthetic-injection metrics
python evaluation/rag_evaluation.py       # retriever: Top-1 accuracy, Top-3 recall, MRR
```

HDFS_2k has no ground-truth failure labels, so both scripts are explicit
about what they can and can't measure - see
[`docs/evaluation.md`](docs/evaluation.md) for real numbers from real runs,
including a case study where the model's top-3 highest-scored windows
exactly matched three planted incidents in a synthetic test log.

## Docker

```bash
docker build -t hdfs-incident-agent .
docker run -p 8000:8000 \
  -v "$(pwd)/data:/app/data" -v "$(pwd)/models:/app/models" -v "$(pwd)/vector_db:/app/vector_db" \
  --env-file .env \
  hdfs-incident-agent
```

Data, models, and the vector index are mounted as volumes rather than baked
into the image, since they're large and regenerable via `scripts/`. See the
`Dockerfile` for running the Streamlit UI from the same image.

## Configuration

All configuration is environment-variable driven - see
[`.env.example`](.env.example) for the full list (`HDFS_RAW_DIR`,
`HDFS_MODEL_PATH`, `HDFS_INDEX_PATH`, `HDFS_WINDOW`, `HDFS_USE_LLM`,
`HUGGINGFACEHUB_API_TOKEN`, `HDFS_API_URL`). Every one has a sensible
repo-relative default, so nothing needs to be set for local development.

## Tech stack

| Purpose | Technology |
|---|---|
| Data | Pandas |
| Anomaly detection | scikit-learn (Isolation Forest) |
| Explainability | SHAP |
| Embeddings + retrieval | LangChain, HuggingFace, FAISS, BM25 |
| API | FastAPI |
| UI | Streamlit |
| Testing | Pytest |
| Deployment | Docker, GitHub Actions |

## Known limitations

- "Abnormal" means unusual relative to this dataset, not verified as a real
  failure - HDFS_2k has no ground truth to check against.
- The 5 starter runbooks in `data/knowledge_base/` are a starting point;
  expand them with real operational knowledge before relying on this.
- Recommended actions come from general HDFS operational knowledge, not a
  validated internal playbook.

Full discussion in [`docs/architecture.md`](docs/architecture.md#known-limitations)
and [`docs/evaluation.md`](docs/evaluation.md#honest-limitations).
