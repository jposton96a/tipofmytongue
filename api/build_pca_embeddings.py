import sys
from pymilvus import (
    connections,
    utility,
    Collection,
    MilvusException
)

from app.milvus_utils import create_milvus_collection, insert_pca_embeddings_in_milvus



def main(
    path_to_words,
    embedding_dims,
    batch_size,
    pca_collection_name,
    embedding_collection_name,
    pca_model_path,
    milvus_uri
):
    # Establish connection to Milvus
    try:
        connections.connect(alias="default", uri=milvus_uri)
        embedding_collection = Collection(embedding_collection_name)
        embedding_collection.load()
    except MilvusException as e:
        print(f"Could not establish connection to Milvus: {e}")
        sys.ext(0)

    if not utility.has_collection(pca_collection_name):
        pca_collection = create_milvus_collection(pca_collection_name, embedding_dims)
        insert_pca_embeddings_in_milvus(
            path_to_words,
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
                    path_to_words,
                    batch_size,
                    pca_collection,
                    embedding_collection,
                    pca_model_path
                )

            case "2": # If pick up from num_entities index
                insert_pca_embeddings_in_milvus(
                    path_to_words,
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
        path_to_words="res/words.txt",
        embedding_dims=3,
        batch_size=5000,
        pca_collection_name="tipofmytongue_pca",
        embedding_collection_name="tipofmytongue",
        pca_model_path="res/pca_transform.pkl",
        milvus_uri="grpc://localhost:19530"
    )