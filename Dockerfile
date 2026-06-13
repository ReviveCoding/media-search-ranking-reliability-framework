FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install --no-cache-dir .
COPY scripts ./scripts
COPY configs ./configs
ENV PYTHONUNBUFFERED=1
CMD ["media-search-run", "--config", "configs/pipeline.yaml", "--mode", "demo"]
