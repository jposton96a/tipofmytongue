FROM python:3.10.13-slim-bullseye
WORKDIR /src

COPY pyproject.toml ./

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends curl \
    && pip install --no-cache-dir poetry \
    && poetry config virtualenvs.create false \
    && poetry install --with docker \
    --verbose --no-cache --no-interaction --no-ansi \
    && rm -rf ~/.cache/pypoetry/artifacts

COPY main.py ./
COPY app/*.py app/

CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8100" ]