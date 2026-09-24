# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.12
ARG POETRY_VERSION=2.3.2

FROM python:${PYTHON_VERSION}-slim AS builder
ARG POETRY_VERSION

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1

RUN pip install "poetry==${POETRY_VERSION}"

WORKDIR /app

# Dependencies first: this layer is only rebuilt when the lock file changes.
COPY pyproject.toml poetry.lock README.md ./
RUN poetry install --only main --no-root

COPY src ./src
RUN poetry install --only main


FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app --shell /usr/sbin/nologin app

WORKDIR /app

COPY --from=builder --chown=app:app /app/.venv ./.venv
COPY --from=builder --chown=app:app /app/src ./src

USER app

# The API keys are never baked into the image: pass them at run time
# (docker run --env-file .env ...).
ENTRYPOINT ["pychat"]
CMD ["chat"]
