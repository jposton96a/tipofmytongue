FROM python:3.10.13-slim-bullseye
WORKDIR /src

COPY res/words.txt res/
COPY res/word_embeddings_cache.npz.chk_non_norm_466503 res/

COPY poetry.lock pyproject.toml .
RUN pip install "poetry==1.6.1" && \
    poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi

COPY *.py .
COPY app/*.py app/

CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000" ]