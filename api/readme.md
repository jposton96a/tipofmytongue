# Tip of My Tongue | Backend API Server

## Overview
This directory contains the scripts & API implementation used to run the backend for [TipOfMyTongue](../readme.md). This server relies on Milvus Vector Store and Triton Inference Server deployed with docker compose.

### Dependencies
The server relies on a list of words & a pre-computed PCA model file. Each of these must be created before starting the server:

1. Vocabulary list `res/words.txt` - (created in Step #1) a list of words to query against
2. PCA Transform Model `res/pca_model_<MODEL_NAME>.pkl` - (created in Step #6) the trained PCA model

## Development Setup

1. Download vocabulary:
    ```bash
    wget https://raw.githubusercontent.com/dwyl/english-words/master/words.txt -O res/words.txt
    ```

2. Setup Python dependencies:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install poetry
    poetry install --with local
    ```
    Poetry will install the "local" group for packages running in the venv, and the "docker" group for packages in the Docker container.

3. Prepare Triton by creating an `.env` file with AWS credentials and the model information in the `docker-compose.yml`:
    ```bash
    cd ../triton
    touch .env
    ```

    ```bash title=".env"
    # triton/.env
    AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID>
    AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY>
    AWS_DEFAULT_REGION=<AWS_DEFAULT_REGION>
    ```

    ```bash
    # docker-compose.yml
    services:
      triton:
        environment:
          MODEL: all-MiniLM-L6-v2
          MODEL_REPO: s3://tipofmytongue-models-gpu
    ```
    There are other options for models. For more information see the Triton [readme](../triton/readme.md).

4. Start Milvus and Triton services (Triton depends on Milvus, so Milvus will start when Triton is launched):
    ```bash
    cd ..

    docker compose up -d triton
    ```
    If you have issues, remove the `-d` flag to troubleshoot.

5. Build the embedding cache into Milvus (this step will take the longest ~5-80 minutes depending on hardware and model dimensions). In steps 5-8, set the model name and model dimensions manually in `main()`.
    ```bash
    cd api

    # Options can be changed at the bottom of this file
    python build_embeddings.py
    ```

6. Build the reduced dimensional PCA embeddings and save the fitted model (this step should take a couple minutes):
    ```bash
    # Options can be changed at the bottom of this file
    python build_pca_embeddings.py
    ```
    By default, the pickle model will save to `res/pca_model_<MODEL_NAME>.pkl`.

7. (Optional - debug tooling) Query an embedding against the cache db:
    ```bash
    # Options can be changed at the bottom of this file
    python query_words.py
    ```

8. Visualize similarity word embeddings. Open the file and set your own `input_word`:
    ```bash
    # Options can be changed at the bottom of this file
    # As a side effect, this renders all processed embeddings into `plot.png`
    python visualize_embeddings.py
    ```

## Deployment

1. Set the model name in the `docker-compose.yml` and run the API server ():
    ```bash
    # docker-compose.yml
    services:
      app:
        environment:
          MODEL: all-MiniLM-L6-v2
    ```

    ```bash
    docker compose up -d
    ```

2. Access the frontend at http://localhost:8080

3. Shutdown the server with:
    ```bash
    docker compose down
    ```

### Resources

##### Karpathy's KNN vs SVM
https://github.com/karpathy/randomfun/blob/master/knn_vs_svm.ipynb

##### Deploy FastAPI on Lambda
https://mangum.io/
https://www.deadbear.io/simple-serverless-fastapi-with-aws-lambda/
https://ademoverflow.com/blog/tutorial-fastapi-aws-lambda-serverless/

## Issues

* There have been issues with installing packages with poetry. If your poetry installation is stuck at "resolving dependencies..." add the verbose flag "-vvv". Furthermore, if it is stuck at "Loading MacOS", add the following variable to your environment by running this:
    ```bash
    export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
    ```