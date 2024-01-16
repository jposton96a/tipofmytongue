import os
import sys
import joblib
from typing import List

from enum import Enum
from mangum import Mangum
from pydantic import BaseModel
from fastapi import FastAPI, status
from contextlib import asynccontextmanager
from pymilvus import connections, Collection, MilvusException

from app.embedding_utils import create_embedding
from app.query_utils import k_similar_words, query_pca_word_embedding
from app.triton_utils import TritonRemoteModel



###########################
### App Dependencies
###########################
model_name = os.getenv("MODEL_NAME")
milvus_uri = "grpc://standalone:19530"
triton_uri = "grpc://triton:8001"
pca_model_dir = "res/"

embedding_collection_name = model_name.replace("-", "_") if "-" in model_name else model_name
pca_collection_name = embedding_collection_name + "_pca"
# append model name to PCA model to allow for more than one
pca_model_path = os.path.join(pca_model_dir, "pca_model_" + embedding_collection_name + ".pkl")



# Establish connection to Milvus and Triton service
try:
    connections.connect(alias="default", uri=milvus_uri)
    model = TritonRemoteModel(uri=triton_uri, model_name=model_name)
except MilvusException as e:
    print(f"Could not establish connection to Milvus: {e}")
    sys.exit(0)
except ConnectionRefusedError as e:
    print(f"Could not establish connection to Triton: {e}")
    sys.exit(0)

# Load Milvus collections for embeddings and pca_embeddings
try:
    embedding_collection = Collection(embedding_collection_name)
    embedding_collection.load()
except Exception as e:
    print(f"Milvus collection load error: {e}.")
    print(f"Make sure the collection `{embedding_collection_name}` exists and is populated.")
    sys.exit(0)
try:
    pca_collection = Collection(pca_collection_name)
    pca_collection.load()
except Exception as e:
    print(f"Milvus collection load error: {e}.")
    print(f"Make sure the collection `{pca_collection_name}` exists and is populated.")
    sys.exit(0)

# Load fitted PCA model
try:
    transform_model = joblib.load(pca_model_path)
except (FileNotFoundError, IndexError) as e:
    print(f"Error loading PCA model: {e}")
    print("Make sure you have defined a model and the file path is correct.")
    print("Run build_pca_embeddings.py to create and fit a PCA model.")
    sys.exit(0)



###########################
### REST API
###########################
# Define dictionary for global variables
global_data = dict()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Init
    global_data["main_embedding"] = []
    global_data["search_vectors"] = []
    global_data["result_vectors"] = []
    yield

    # Shutdown
    global_data.clear()

app = FastAPI(lifespan=lifespan)

class HealthCheck(BaseModel):
    status: str = "OK"

class Function(str, Enum):
    start = 'start'
    more_like = "more_like"
    less_like = "less_like"

class Operation(BaseModel):
    id: str | None = None
    function: Function
    description: str
    results: list | None = None
    selected_words: list | None = None


@app.get("/health", status_code=status.HTTP_200_OK, response_model=HealthCheck)
def health():
    return HealthCheck(status="OK")

@app.post("/operations")
async def create_operation(ops: List[Operation]):
    if len(ops) == 1:
        global_data["main_embedding"] = []
    op = ops[-1]

    match op.function:
        case Function.start:
            global_data["main_embedding"].append(create_embedding(op.description, model))
        case Function.less_like:
            global_data["main_embedding"][0] -= create_embedding(op.description, model)
        case Function.more_like:
            global_data["main_embedding"][0] += create_embedding(op.description, model)

    similar_words = k_similar_words(global_data["main_embedding"][0], embedding_collection, pca_collection)
    ops[-1].results = [{"word": word[0], "dist": word[1], "id": word[2]} for word in similar_words]

    return ops

@app.post("/scatter")
async def scatter(ops: List[Operation]):
    # reset vectors after reset
    if len(ops) == 1:
        global_data["search_vectors"] = []
        global_data["result_vectors"] = []

    op = ops[-1]
    transformed_search_vector = transform_model.transform(global_data["main_embedding"][0].reshape(1, -1))
    global_data["search_vectors"].append({
        "description": op.description,
        "coords": transformed_search_vector[0].tolist()
    })

    if op.results:
        ids, words = [res["id"] for res in op.results], [res["word"] for res in op.results]
        response = query_pca_word_embedding(ids, pca_collection)
        for i in range(len(ids)):
            global_data["result_vectors"].append({
                "description": words[i],
                "coords": response[i]
            })

    response = {
        "search_vectors": global_data["search_vectors"],
        "result_vectors": global_data["result_vectors"]
    }

    return response

handler = Mangum(app, lifespan="on")