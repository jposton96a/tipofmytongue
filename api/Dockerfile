FROM python:3.10.13-slim-bullseye
WORKDIR /src

COPY pyproject.toml ./

RUN apt update -y \
    && apt install -y curl \
    && pip install "poetry==1.6.1" \
    && poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --with docker \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY main.py ./
COPY app/*.py app/

CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8100" ]