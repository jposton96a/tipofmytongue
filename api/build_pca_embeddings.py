from pymilvus import (
    connections,
    utility,
    Collection,
    MilvusException
)

from app.milvus_utils import create_milvus_collection, insert_pca_embeddings_in_milvus



if __name__ == "__main__":
    embedding_dims = 3
    batch_size = 5000
    collection_name = "tipofmytongue_pca"
    path_to_words = "res/words.txt"
    pca_model_path = "res/pca_transform.pkl"

    # Establish connection to Milvus and Triton
    try:
        connections.connect(alias="default", host="localhost", port="19530") # Milvus
        embedding_collection = Collection("tipofmytongue")
        embedding_collection.load()
    except MilvusException as e:
        print(f"Could not establish connection to Milvus: {e}")
        raise

    if not utility.has_collection(collection_name):
        collection = create_milvus_collection(collection_name, embedding_dims)
        insert_pca_embeddings_in_milvus(
            path_to_words,
            batch_size,
            collection,
            embedding_collection,
            pca_model_path
        )

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
                insert_pca_embeddings_in_milvus(
                    path_to_words,
                    batch_size,
                    collection,
                    embedding_collection,
                    pca_model_path
                )

            case "2": # If pick up from num_entities index
                insert_pca_embeddings_in_milvus(
                    path_to_words,
                    batch_size,
                    collection,
                    embedding_collection,
                    pca_model_path,
                    num_entities
                )

            case "3": # Remove existing Milvus collection
                utility.drop_collection(collection_name)
                print(f"Successfully remove collection `{collection_name}`")

            case "4": # Exit program
                exit()

            case _:
                print("Invalid user input, select a number from the options above.")

    connections.disconnect("default")