import math
import pymilvus
import numpy as np

from tqdm import tqdm
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    MilvusException
)

from app.triton_utils import TritonRemoteModel



def insert_entries_in_milvus(path_to_words, batch_size, model, collection, start_index=0, flush_every=10000):
    # flushing basically "saves" the Milvus collection
    # It is time consuming, so we only flush every 20000 entities added
    flush = math.ceil(flush_every / batch_size)

    # read all words into list
    with open(path_to_words, "r") as file:
        words = file.read().splitlines()

    num_words = len(words)

    for flush_index, batch in enumerate(tqdm(
        range(start_index, num_words, batch_size),
        desc=f"Batch number with batch size of {batch_size}",
        ncols=150,
        bar_format="{l_bar}{bar:10}{r_bar}"
    )):
        text = words[batch:batch+batch_size]

        if len(text) != batch_size: # If last batch doesn't match the normal batch_size
            last_text = text.copy()
            new_batch_size = batch_size - len(last_text)
            model_output = model(new_batch_size, np.array([str.encode(word) for word in last_text]))
        else:
            model_output = model(batch_size, np.array([str.encode(word) for word in text]))

        insert_result = collection.insert([
            text,
            model_output.numpy()
        ])

        if flush_index % flush == 0: # Save, or "flush", Milvus collection every 20,000 entities added
            collection.flush()
    
    collection.flush()
    print(f"Number of entities in Milvus: {collection.num_entities}")
    return


def create_milvus_collection(embedding_dims, collection_name):
    fields = [
        FieldSchema(name="word", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dims)
    ]
    schema = CollectionSchema(
        fields=fields,
        description="Schema describing the Milvus collection, a vector store for the word embeddings."
    )
    return Collection(name=collection_name, schema=schema, using="default", consistency_level="Strong")



if __name__ == "__main__":
    # Establish connection to Milvus and Triton
    try:
        connections.connect(alias="default", host="localhost", port="19530") # Milvus
        model = TritonRemoteModel("http://localhost:8100", "gte-large")      # Triton
    except MilvusException as e:
        print(f"Could not establish connection to Milvus: {e}")
    except ConnectionRefusedError as e:
        print(f"Could not establish connection to Triton: {e}")

    embedding_dims = 1024
    batch_size = 100
    collection_name = "tipofmytongue"
    path_to_words = "res/words.txt"

    if not utility.has_collection(collection_name):
        collection = create_milvus_collection(embedding_dims, collection_name)
        insert_entries_in_milvus(path_to_words, batch_size, model, collection)

    else:
        collection = pymilvus.Collection(collection_name)
        num_entities = collection.num_entities
        print(f"The collection `{collection_name}` already exists with {num_entities} entries.")

        user_input = input(
            f"""
            Enter a number for your option:
            1. Overwrite existsing Milvus collection
            2. Pick up from entry {num_entities}
            3. Remove existing Milvus collection
            4. Exit program
            """
        )
        
        match user_input:
            case "1": # If overwrite, remove old collection and start over
                utility.drop_collection(collection_name)
                collection = create_milvus_collection(embedding_dims, collection_name)
                insert_entries_in_milvus(path_to_words, batch_size, model, collection)

            case "2": # If pick up from num_entities index
                insert_entries_in_milvus(path_to_words, batch_size, model, collection, num_entities)

            case "3": # Remove existing Milvus collection
                utility.drop_collection(collection_name)
                print(f"Successfully remove collection `{collection_name}`")

            case "4": # Exit program
                exit()

            case _:
                print("Invalid user input, select a number from the options above.")
