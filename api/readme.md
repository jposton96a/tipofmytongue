# Tip of My Tongue | Backend API Server

## Overview
This directory contains the scripts & API implementation used to run the backend for [TipOfMyTongue](../readme). This server relies on Milvus Vector Store and Triton Inference Server deployed with docker compose.

### Dependencies
The server relies on a list of words & a pre-computed PCA model file. Each of these must be created before starting the server:

1. Vocabulary list `res/words.txt` - (created in Step #1) a list of words to query against
2. PCA Transform Model `res/pca_transform.pkl` - (created in Step #6) the trained PCA model

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
    poetry install

    pip install optimum[onnxruntime]
    ```
3. Prepare Triton by downloading the model and converting it to ONNX format:
    ```bash
    cd app/
    python triton_utils.py
    mv model/model_quantized.onnx ../../triton/transformer/1/model.onnx
    rm -rf model/
    ```
    Triton should already be prepared after this step, if you have trouble look in the `triton/` directory. For more information about Triton see [Triton readme](../triton/readme.md).

4. Start Milvus and Triton services (triton-server will need time to build):
    ```bash
    cd ..
    docker compose up -d etcd minio standalone triton
    ```
    If you have issues, remove the `-d` flag to troubleshoot.

5. Build the embedding cache into Milvus  (this step will take the longest ~20-80 minutes depending on hardware):
    ```bash
    python build_embeddings.py
    ```

6. Build the reduced dimensional PCA embeddings and save the fitted model (this step shouldn't take too long):
    ```bash
    python build_pca_embeddings.py
    ```
    By default, the pickle model will save to `res/pca_transform.pkl`.

7. (Optional - debug tooling) Query an embedding against the cache db:
    ```bash
    # References the embedding cache @ res/<embedding_cache_name>.npz
    python query_words.py
    ```

8. Visualize similarity word embeddings. Open the file and set your own `input_word`:
    ```bash
    # as a side effect, this renders all processed embeddings into `plot.png` 
    python visualize_embeddings.py
    ```
## Deployment

1. Run the API server ()
    ```bash
    cd ..

    # If Milvus and Triton are still running
    docker compose up -d app ui_server

    # If Milvus and Triton are not running, run all services:
    docker compose up -d
    ```

2. Access the frontend at http://localhost:8080

### Resources

##### Karpathy's KNN vs SVM
https://github.com/karpathy/randomfun/blob/master/knn_vs_svm.ipynb

##### Deploy FastAPI on Lambda
https://mangum.io/
https://www.deadbear.io/simple-serverless-fastapi-with-aws-lambda/
https://ademoverflow.com/blog/tutorial-fastapi-aws-lambda-serverless/