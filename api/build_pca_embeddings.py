import os
import sys

from pymilvus import (
    connections,
    utility,
    Collection,
    MilvusException
)

from app.milvus_utils import create_milvus_collection, insert_pca_embeddings_in_milvus



def main(
    model_name,
    embedding_dims,
    batch_size,
    path_to_vocab,
    pca_model_dir,
    milvus_uri,
    connection_timeout=10
):
    # Milvus doesn't allow hyphens, so replace with underscores
    embedding_collection_name = model_name.replace("-", "_") if "-" in model_name else model_name
    pca_collection_name = embedding_collection_name + "_pca"
    # append model name to PCA model to allow for more than one
    pca_model_path = os.path.join(pca_model_dir, "pca_model_" + embedding_collection_name + ".pkl")

    # Establish connection to Milvus
    try:
        connections.connect(alias="default", uri=milvus_uri, timeout=connection_timeout)
        embedding_collection = Collection(embedding_collection_name)
        embedding_collection.load()
    except MilvusException as e:
        print(f"Could not establish connection to Milvus: {e}")
        sys.ext(0)

    if not utility.has_collection(pca_collection_name):
        pca_collection = create_milvus_collection(pca_collection_name, embedding_dims)
        insert_pca_embeddings_in_milvus(
            path_to_vocab,
            batch_size,
            pca_collection,
            embedding_collection,
            pca_model_path
        )

    else:
        pca_collection = Collection(pca_collection_name)
        num_entities = pca_collection.num_entities
        print(f"The collection `{pca_collection_name}` already exists with {num_entities} entries.")

        user_input = input(
            f"""\
            \nEnter a number for your option:\
            \n  1. Overwrite existsing Milvus collection\
            \n  2. Pick up from entry {num_entities}\
            \n  3. Remove existing Milvus collection\
            \n  4. Exit program\
            \n"""
        )
        
        match user_input:
            case "1": # If overwrite, remove old collection and start over
                utility.drop_collection(pca_collection_name)
                pca_collection = create_milvus_collection(pca_collection_name, embedding_dims)
                insert_pca_embeddings_in_milvus(
                    path_to_vocab,
                    batch_size,
                    pca_collection,
                    embedding_collection,
                    pca_model_path
                )

            case "2": # If pick up from num_entities index
                insert_pca_embeddings_in_milvus(
                    path_to_vocab,
                    batch_size,
                    pca_collection,
                    embedding_collection,
                    pca_model_path,
                    num_entities
                )

            case "3": # Remove existing Milvus collection
                utility.drop_collection(pca_collection_name)
                print(f"Successfully removed collection `{pca_collection_name}`")

            case "4": # Exit program
                exit()

            case _:
                print("Invalid user input, select a number from the options above.")

    connections.disconnect("default")



if __name__ == "__main__":
    main(
        model_name="gte-small",
        embedding_dims=3,
        batch_size=5000,
        path_to_vocab="res/words.txt",
        pca_model_dir="res/",
        milvus_uri="grpc://localhost:19530"
    )