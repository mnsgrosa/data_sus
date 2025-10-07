FROM python:3.11.7-slim

ENV PATH ="${VENV_PATH}/bin:${PATH}" \
    VENV_PATH="/opt/venv"

RUN pip install uv

RUN apt-get update && apt-get install -y libgl1-mesa-glx

RUN uv venv --python=3.11.7 ${VENV_PATH}

# Copy only requirements to cache them in docker layer
WORKDIR /etl
COPY uv.lock pyproject.toml /etl/

ENV UV_PROJECT_ENVIRONMENT="/usr/local/"

RUN uv sync --locked $(test "$ENV" == prod && echo "--no-dev")

COPY . /etl/


RUN mkdir -p /etl/data && chmod -R 777 /etl/data


RUN uv pip install -e . --system