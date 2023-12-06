import pymilvus
import math
import torch
import numpy as np

from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection
)

from app.embedding_utils import TritonRemoteModel


def save_embeddings(file_path, embeddings):
    np_embeddings = embeddings

    # save the embeddings array to the .npz file using a keyword argument
    np.savez(file_path, embeddings=np_embeddings)
    del(np_embeddings)

def count_populated(a):
    for i, line in enumerate(a):
        if line.nonzero()[0].size == 0 or np.any(np.isnan(line)):
            return i

connections.connect(alias="default", host="localhost", port="19530")

dim = 1024
batch_size = 16
collection_name = 'tiptest'

collection_exists = utility.has_collection(collection_name)

if not collection_exists:
    fields = [
        FieldSchema(name="word", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    schema = CollectionSchema(fields=fields, description="Schema describing the Milvus collection, a vector store for the word embeddings.")
    tiptest = Collection(name="tiptest", schema=schema, using="default", consistency_level="Strong")

    # open the file that contains the words
    with open("res/words.txt", "r") as file:
        words = file.read().splitlines()

    num_words = len(words)
    batches = math.ceil(num_words / batch_size)

    model = TritonRemoteModel("http://localhost:8100", "gte-large")

    for batch in range(0, batches, batch_size):
        text = words[batch:batch+batch_size]

        model_output = model(np.array([str.encode(word) for word in text]))

        insert_result = tiptest.insert([
            text,
            model_output.numpy()
        ])
    
    tiptest.flush()
    print(f"Number of entities in Milvus: {tiptest.num_entities}")
else:
    utility.drop_collection("tiptest")
    print("exists")
