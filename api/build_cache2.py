import pymilvus
import numpy as np

from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
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
    file = open("res/words.txt", "r")
    lines = file.readlines()
    file.close()

    model = TritonRemoteModel("http://localhost:8100", "gte-large")

    embeddings = np.empty((len(lines), 1024))
    for i, line in enumerate(lines):
        text = line.rstrip("\n")

        model_output = model(np.array([str.encode(text)]))

        embedding = np.array(model_output)
        embeddings[i] = embedding
    
    insert_result = tiptest.insert(embeddings)
    tiptest.flush()
    print(f"Number of entities in Milvus: {tiptest.num_entities}")
else:
    print("exists")