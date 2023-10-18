# Tip of My Tongue (.ai)

## Overview
Tip of My Tongue is a new approach to the thesaurus. Where a thesaurus enables a user to search for synonyms of a given word, this application allows users to describe the word they're looking for & find semantically similar words. The (future) interface enables the user to navigate the embedding vector space to hone their results in to the word they're looking for.

Currently, the the interface for finding words is through the logic in `query_words.py`.

## Development Setup

1. Download Vocabulary
    ```bash
    wget https://raw.githubusercontent.com/dwyl/english-words/master/words.txt -O res/words.txt
    ```
2. Setup Python dependencies
    ```bash
    source .venv/bin/activate
    pip install poetry
    poetry install 
    ```

3. Build embedding cache (Takes a while! Save the result)
    ```bash
    # Creates res/<embedding_cache_name>.npz
    export OPENAI_API_KEY="<Insert OPENAI Key>"
    python src/build_cache.py
    ```
    
    **_#TODO_** Improve build stage. Currently the process took me ~36hrs to embed the entire vocab list.
    - [ ] Parallelize embeddings vocab
    - [ ] Support local embedding operations

4. (Optional - debug tooling) Query an embedding against the cache db
    ```bash
    # References the embedding cache @ res/<embedding_cache_name>.npz
    python src/query_words.py
    ```

5. Run the API server
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload --env-file config.env
    ```

### Resources
https://www.deadbear.io/simple-serverless-fastapi-with-aws-lambda/
https://mangum.io/
https://ademoverflow.com/blog/tutorial-fastapi-aws-lambda-serverless/