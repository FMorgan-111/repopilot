FROM python:3.12-slim

LABEL org.opencontainers.image.title="RepoPilot"
LABEL org.opencontainers.image.description="AI-powered GitHub issue → fix PR, with self-reflective agent loop"
LABEL org.opencontainers.image.url="https://github.com/FMorgan-111/repopilot"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir .

ENV GIT_AUTHOR_NAME="RepoPilot"
ENV GIT_AUTHOR_EMAIL="repopilot@local"
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["repopilot"]
