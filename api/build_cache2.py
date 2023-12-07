import math
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

from app.embedding_utils import TritonRemoteModel

try:
    connections.connect(alias="default", host="localhost", port="19530")
    model = TritonRemoteModel("http://localhost:8100", "gte-large")
except MilvusException as e:
    print(f"Could not establish connection to Milvus: {e}")
except ConnectionRefusedError as e:
    print(f"Could not establish connection to Triton: {e}")

embedding_dims = 1024
batch_size = 16
collection_name = 'tiptest'

if not utility.has_collection(collection_name):
    fields = [
        FieldSchema(name="word", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dims)
    ]
    schema = CollectionSchema(fields=fields, description="Schema describing the Milvus collection, a vector store for the word embeddings.")
    tiptest = Collection(name="tiptest", schema=schema, using="default", consistency_level="Strong")

    # open the file that contains the words
    with open("res/words.txt", "r") as file:
        words = file.read().splitlines()

    num_words = len(words)
    batches = math.ceil(num_words / batch_size)

    for batch in tqdm(
        range(460000, num_words, batch_size),
        desc=f"Batch number with batch size of {batch_size}",
        ncols=150,
        bar_format="{l_bar}{bar:10}{r_bar}"
    ):
        text = words[batch:batch+batch_size]

        if len(text) != batch_size:
            last_text = text.copy()
            L = batch_size - len(last_text)
            for i in range(L):
                last_text.append("filler")

            model_output = model(np.array([str.encode(word) for word in last_text]))[:-L, :]
        else:
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
