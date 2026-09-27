# HDFS Incident Intelligence Agent - API image.
#
# Runs the FastAPI backend (src/api). Note on size: requirements.txt pulls in
# sentence-transformers/torch (for embeddings) and faiss-cpu, so this image
# is several GB - that's inherent to the RAG stack, not something a slimmer
# base image fixes.
#
# Build:
#   docker build -t hdfs-incident-agent .
#
# Run (mount your data/model/index from the host so the image itself stays
# free of large, regenerable artifacts - see .dockerignore):
#   docker run -p 8000:8000 \
#     -v "$(pwd)/data:/app/data" \
#     -v "$(pwd)/models:/app/models" \
#     -v "$(pwd)/vector_db:/app/vector_db" \
#     --env-file .env \
#     hdfs-incident-agent
#
# Run the Streamlit UI from the SAME image instead, against that API:
#   docker run -p 8501:8501 -e HDFS_API_URL=http://host.docker.internal:8000 \
#     hdfs-incident-agent \
#     streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501

FROM python:3.11-slim

# Prevents .pyc files and forces stdout/stderr to be unbuffered, so
# `docker logs` shows output immediately instead of being buffered.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first (separate layer) so code changes don't bust the
# pip-install cache - this is the single biggest build-time cost here.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Now the application code.
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY app/ ./app/
COPY data/knowledge_base/ ./data/knowledge_base/

# data/raw, data/processed, models/, vector_db/ are intentionally NOT copied
# in (see .dockerignore) - mount them as volumes at `docker run` time. Empty
# placeholders here just let Settings()'s default paths resolve without error
# before anything is mounted.
RUN mkdir -p data/raw/hdfs data/processed models vector_db

# Runs as a non-root user - standard container hardening practice.
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Liveness check: confirms the API process is up and answering, not that
# every component (model/index) is loaded - GET /health always returns 200
# and reports per-component status in its body for that finer-grained check.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]