import sys
from pymilvus import (
    connections,
    utility,
    Collection,
    MilvusException
)

from app.triton_utils import TritonRemoteModel
from app.milvus_utils import create_milvus_collection, insert_embeddings_in_milvus



def main(
    path_to_words,
    embedding_dims,
    batch_size,
    collection_name,
    milvus_uri,
    triton_uri
):
    # Establish connection to Milvus and Triton
    try:
        connections.connect(alias="default", uri=milvus_uri)
        model = TritonRemoteModel(url=triton_uri, model_name="gte-large")
    except MilvusException as e:
        print(f"Could not establish connection to Milvus: {e}")
        sys.exit(0)
    except ConnectionRefusedError as e:
        print(f"Could not establish connection to Triton: {e}")
        sys.exit(0)

    if not utility.has_collection(collection_name):
        collection = create_milvus_collection(collection_name, embedding_dims)
        insert_embeddings_in_milvus(path_to_words, batch_size, model, collection)

    else:
        collection = Collection(collection_name)
        num_entities = collection.num_entities
        print(f"The collection `{collection_name}` already exists with {num_entities} entries.")

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
                utility.drop_collection(collection_name)
                collection = create_milvus_collection(collection_name, embedding_dims)
                insert_embeddings_in_milvus(path_to_words, batch_size, model, collection)

            case "2": # If pick up from num_entities index
                insert_embeddings_in_milvus(path_to_words, batch_size, model, collection, num_entities)

            case "3": # Remove existing Milvus collection
                utility.drop_collection(collection_name)
                print(f"Successfully removed collection `{collection_name}`")
                connections.disconnect("default")

            case "4": # Exit program
                exit()

            case _:
                print("Invalid user input, select a number from the options above.")

    connections.disconnect("default")



if __name__ == "__main__":
    main(
        embedding_dims=1024,
        batch_size=64,
        collection_name="tipofmytongue",
        path_to_words="res/words.txt",
        milvus_uri="grpc://localhost:19530",
        triton_uri="grpc://localhost:8001"
    )