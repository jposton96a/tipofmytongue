import os
import sys

import joblib
import numpy as np
import matplotlib.pyplot as plt
from pymilvus import connections, Collection, MilvusException

from app.embedding_utils import create_embedding
from app.query_utils import k_similar_words
from app.triton_utils import TritonRemoteModel



def main(
    input_word,
    num_points,
    model_name,
    pca_model_dir,
    milvus_uri,
    triton_uri
):
    # Milvus doesn't allow hyphens, so replace with underscores
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

    embedding_collection = Collection(embedding_collection_name)
    embedding_collection.load()

    pca_collection = Collection(pca_collection_name)
    pca_collection.load()

    try:
        transform_model = joblib.load(pca_model_path)
    except (FileNotFoundError, IndexError) as e:
        print(f"Error loading PCA model: {e}")
        print("Make sure you have defined a model and the file path is correct.")
        print("Run build_pca_embeddings.py to create and fit a PCA model.")
        sys.exit(0)


    # Create input vector and 100 similar word embeddings to plot
    input_vector = create_embedding(input_word, model)
    similar_words = k_similar_words(input_vector, embedding_collection, pca_collection, k=num_points)
    words = [word[0] for word in similar_words]
    reduced_embeddings = np.array([word[1] for word in similar_words])
    transformed_search_vector = transform_model.transform(input_vector.reshape(1, -1))[0]

    # Create a 3D scatter plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    x, y, z = reduced_embeddings[:, 0], reduced_embeddings[:, 1], reduced_embeddings[:, 2]
    ax.scatter(x, y, z)

    # Plot the search vector
    ax.scatter(transformed_search_vector[0], transformed_search_vector[1], transformed_search_vector[2], color='red', label='Search Vector')

    # Label each point with its corresponding word
    for i, word in enumerate(words):
        ax.text(x[i], y[i], z[i], word, size=10, zorder=1, color='k')

    ax.set_title(f"3D PCA similarity word embeddings for '{input_word.upper()}'")
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.set_zlabel("PCA3")
    plt.savefig("plot.png")
    print("Results save to `plot.png`")



if __name__ == "__main__":
    main(
        input_word="aliens",
        num_points=50,
        model_name="gte-small",
        pca_model_dir="res/",
        milvus_uri="grpc://localhost:19530",
        triton_uri="grpc://localhost:8001"
    )