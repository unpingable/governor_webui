# Governor WebUI
# OpenAI-compatible API with switchable backends and Governor integration

FROM python:3.11-slim

WORKDIR /app

# Install agent-governor from local source (not on PyPI)
COPY agent-governor/ /tmp/agent-governor/
RUN pip install --no-cache-dir /tmp/agent-governor/ && rm -rf /tmp/agent-governor/

# Install receipt-v1 from local source (not on PyPI)
COPY receipt-v1/ /tmp/receipt-v1/
RUN pip install --no-cache-dir /tmp/receipt-v1/ && rm -rf /tmp/receipt-v1/

# Install dependencies (README.md required by pyproject.toml)
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir .

# Copy source
COPY src/ src/

# Install the package
RUN pip install -e .

# Verify all local-only deps are importable (fail build, not first request)
RUN python3 -c "import governor; import receipt_v1; import gov_webui"

# Entrypoint script: start governor daemon, then uvicorn
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run daemon + adapter
CMD ["/app/entrypoint.sh"]
