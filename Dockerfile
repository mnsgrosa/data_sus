FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_TOOL_BIN_DIR=/usr/local/bin

WORKDIR /app

RUN addgroup --system nonroot && \
    adduser --system --ingroup nonroot --home /home/nonroot nonroot && \
    mkdir -p /home/nonroot/.cache && \
    chown -R nonroot:nonroot /app /home/nonroot

COPY --chown=nonroot:nonroot uv.lock pyproject.toml README.md /app/

COPY --chown=nonroot:nonroot src /app/src

USER nonroot

RUN uv sync --no-cache

COPY --chown=nonroot:nonroot . /app/

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src:$PYTHONPATH"

ENTRYPOINT []
