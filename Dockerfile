# Governor WebUI
# OpenAI-compatible API with switchable backends and Governor integration

FROM python:3.11-slim

WORKDIR /app

# Install dependencies (README.md required by pyproject.toml)
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir .

# Copy source
COPY src/ src/

# Install the package
RUN pip install -e .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the adapter
CMD ["uvicorn", "gov_webui.adapter:app", "--host", "0.0.0.0", "--port", "8000"]
