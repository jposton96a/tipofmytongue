import os
import json
from typing import List

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from enum import Enum
from mangum import Mangum

from app.embedding_utils import create_embedding, load_embeddings, load_word_dicts, count_populated, load_models, TritonRemoteModel
from app.query_utils import find_similar_words
from app.download_embeddings import download_embeddings
from app.transform_utils import create_or_load_transform


###########################
### App Dependencies
###########################

cache_path = "res/word_embeddings_cache.npz"
dict_path = "res/words.txt"

transform_model_path = "res/pca_transform.pkl"
transformed_embeddings_path = "res/pca_transformed_embeddings.npy"

# Download the embedding cache if it doesn't exist locally
if 'DOWNLOAD_CACHE_NAME' in os.environ and not os.path.exists(cache_path):
    cache_name=os.getenv("DOWNLOAD_CACHE_NAME")
    print(cache_name)
    download_embeddings(cache_name)

embeddings = load_embeddings(cache_path)
dictionary = load_word_dicts(dict_path)
print(f"Loaded {len(dictionary)} words from {dict_path}")

# The length of the embeddings will always match the dictionary.
# Some or all of the indexes may be populated
embeddings_count = count_populated(embeddings)
print(f"Loaded {embeddings_count} embeddings from {cache_path}")
if embeddings_count == 0:
    print("Cache empty! Exiting")
    exit(-1)

transform_model, reduced_embeddings = create_or_load_transform(
    embeddings=embeddings, 
    transform_model_path=transform_model_path, 
    transformed_embeddings_path=transformed_embeddings_path
)

word_to_transform_map = dict(zip(dictionary, reduced_embeddings))


# TODO - Remove global dependencies & extract these into query_utils package

def similar_words(q, k=10):
   return find_similar_words(q, embeddings, dictionary, k)

def similar_svn(q, k=10, knn_count=100, c=0.1):
    # Use KNN to find the nearest knn_count words using a basic distance function
    # This helps narrow the search for the SVN, since training an SVN is expensive
    knn_results = similar_words(q, knn_count)
    local_embeddings = [embeddings[x[1]] for x in knn_results]


    from sklearn import svm
    # Append the query into the set of data to evaluate
    x = np.concatenate([q[None,...], local_embeddings])
    y = np.zeros(len(x))
    y[0] = 1 # We have 1 positive sample

    # Train the SVN
    clf = svm.LinearSVC(class_weight='balanced', verbose=False, dual=True, max_iter=10000, tol=1e-6, C=c)
    clf.fit(x, y)
    
    # Only run inference on the knn result embeddings (skip the query included in X)
    similarities = clf.decision_function(local_embeddings)
    sorted_ix = np.argsort(-similarities)

    # Build response format. Mirror the KNN result structure, but include similarity score instead of distance   
    matches = []
    for k in sorted_ix[:k]:
        knn_result_mapping = knn_results[k]
        matches.append((knn_result_mapping[0], knn_result_mapping[1], similarities[k]))
    
    return matches

# Create client connection with Triton Inference Server
model = TritonRemoteModel("http://triton-server:8100", "all-MiniLM-L6-v2")

###########################
### REST API
###########################

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

class Function(str,Enum):
    start = 'start'
    more_like = "more_like"
    less_like = "less_like"

class Operation(BaseModel):
    id: str | None = None
    function: Function
    description: str
    results: list | None = None
    selected_words: list | None = None

def calc_query(ops: List[Operation], tokenizer, model, device):
    q = create_embedding("king", tokenizer, model, device)

    query_vectors = [q]
    for o in ops:
        op_embedding = create_embedding(o.description, tokenizer, model, device)
        match o.function:
            case Function.start:
                q = op_embedding
                query_vectors = []
            case Function.less_like:
                q -= op_embedding
            case Function.more_like:
                q += op_embedding
        
        # Normalize the new embedding
        q = q / np.sqrt((q**2).sum())
        query_vectors.append(q.copy())

    return query_vectors


@app.post("/operations")
async def create_operation(ops: List[Operation]):
    print(ops)

    # TODO: Try "distilbert-base-uncased" model
    tokenizer, device = load_models('sentence-transformers/all-MiniLM-L6-v2')
    query_vectors = calc_query(ops, tokenizer, model, device)
    q = query_vectors[0]

    results = similar_svn(q)
    ops[-1].results = [{"word": x[0], "dist": x[2]} for x in results]

    print(ops)
    return ops

@app.post("/scatter")
async def scatter(ops: List[Operation]):
    print(ops)

    result_vectors = []
    search_vectors = []

    tokenizer, device = load_models('sentence-transformers/all-MiniLM-L6-v2')
    query_vectors = calc_query(ops, tokenizer, model, device)

    for i, o in enumerate(ops):
        search_vector = query_vectors[i]
        transformed_search_vector = transform_model.transform(search_vector.reshape(1, -1))
        search_vectors.append({
            "description": o.description,
            "coords": transformed_search_vector[0].tolist()
        })

        if o.results:
            for r in o.results:
                r_word = r["word"]
                if r_word in word_to_transform_map:
                    result_vectors.append({
                        "description": r_word,
                        "coords": word_to_transform_map[r_word].tolist()
                    })

    response = {
        "search_vectors": search_vectors,
        "result_vectors": result_vectors,
    }
    print(json.dumps(response))
    return response


handler = Mangum(app, lifespan="off")