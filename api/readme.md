# Tip of My Tongue | Backend API Server

## Overview
This directory contains the scripts & API implementation used to run the backend for [TipOfMyTongue](../readme). This server is responsible for querying the embeddings database for words similar to the input query.

### Dependencies
The server relies on a list of words & 3 pre-computed files used for queries. Each of these must be created before starting the server

1. Vocabulary List `res/words.txt` - (created in Step #1) a list of words to query against
1. Embedding Cache `res/word_embeddings_cache.npz` - (created in Step #3) a cache of embeddings for each of the words in vocabulary
1. PCA Transform Model `res/pca_transform.pkl` - (created in Step #5) the trained PCA model
1. PCA Transform Cache `res/pca_transformed_embeddings.npy` - (created in Step #5) a cache of precomputed 3D transforms of the embeddings in the Embedding Cache

## Development Setup

1. Download Vocabulary
    ```bash
    wget https://raw.githubusercontent.com/dwyl/english-words/master/words.txt -O res/words.txt
    ```
2. Setup Python dependencies
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install poetry
    poetry install 
    ```

3. Build embedding cache (Takes a while! Save the result)
    ```bash
    # Creates res/<embedding_cache_name>.npz
    export OPENAI_API_KEY="<Insert OPENAI Key>"
    python build_cache.py
    ```
    
    **_#TODO_** Improve build stage. Currently the process took me ~36hrs to embed the entire vocab list.
    - [ ] Parallelize embeddings vocab
    - [ ] Support local embedding operations

4. (Optional - debug tooling) Query an embedding against the cache db
    ```bash
    # References the embedding cache @ res/<embedding_cache_name>.npz
    python query_words.py
    ```

5. Build the PCA Model
    ```bash
    # as a side effect, this renders all processed embeddings into `plot.png` 
    python visualize_embeddings.py
    ```

6. Run the API server
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload --env-file config.env
    ```

### Resources

##### Karpathy's KNN vs SVM
https://github.com/karpathy/randomfun/blob/master/knn_vs_svm.ipynb

##### Deploy FastAPI on Lambda
https://mangum.io/
https://www.deadbear.io/simple-serverless-fastapi-with-aws-lambda/
https://ademoverflow.com/blog/tutorial-fastapi-aws-lambda-serverless/