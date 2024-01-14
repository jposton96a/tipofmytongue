import joblib
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
from pymilvus import (
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection
)


def create_milvus_collection(collection_name, embedding_dims):
    """
    Create a new Milvus collection for words and word embeddings specifically

    Args:
        collection_name (str): Name for Milvus collection.
        embedding_dims (int): Dimensions of model output.
    
    Returns:
        pymilvus.Collection
    """
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="word", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dims)
    ]
    schema = CollectionSchema(
        fields=fields,
        description="Schema describing the Milvus collection, a vector store for the word embeddings."
    )
    return Collection(name=collection_name, schema=schema, using="default", consistency_level="Strong")


def insert_embeddings_in_milvus(
    path_to_vocab,
    batch_size,
    model,
    collection,
    start_index=0,
    metric_type="L2",
    index_type="IVF_FLAT",
    params={"nlist": 1024}
):
    """
    Insert vector embeddings into a defined Milvus collection

    Args:
        path_to_vocab (str): Path to dictionary text file.
        batch_size (int): The size of the batch sent to Triton for inference.
        model (InferenceServerClient): Defined TritonRemoteModel each word is sent to.
        collection (pymilvus.Collection): Defined Milvus collection for each word, embedding to be sent to.
        start_index (int): Index to start collection insertion on. Defaults to 0 if the collection is empty,
            otherwise will pick up where it left off.
        metric_type (str): Metric used to measure similarity of vectors.
            For floating point vectors: "L2", "IP", "COSINE"
            For binary vectors: "JACCARD", "HAMMING"
        index_type (str): Type of index used to accelerate vector search.
            For floating point vectors: "FLAT", "IVF_FLAT", "IVF_SQ8", "IVF_PQ", "GPU_IVF_FLAT*", "GPU_IVF_PQ*>", "HNSW", "DISKANN*"
            For binary vectors: "BIN_FLAT", "BIN_IVF_FLAT"
        params (dict): See https://milvus.io/docs/index.md

    Returns:
        None
    """
    # read all words into list
    with open(path_to_vocab, "r") as file:
        words = file.read().splitlines()

    num_words = len(words)

    for batch in tqdm(
        range(start_index, num_words, batch_size),
        desc=f"Adding embeddings to Milvus with batch size of {batch_size}",
        ncols=150,
        bar_format="{l_bar}{bar:10}{r_bar}"
    ):
        ran_batching = True
        text = words[batch:batch+batch_size]
        num_text = len(text)

        if num_text != batch_size: # If last batch doesn't match the normal batch_size
            last_text = text.copy()
            model_output = model(num_text, np.array([str.encode(word) for word in last_text])).numpy()
        else:
            model_output = model(batch_size, np.array([str.encode(word) for word in text])).numpy()

        collection.insert([
            [batch + i for i in range(num_text)],
            text,
            model_output
        ])
    
    collection.flush()
    print(f"Number of entities in collection `{collection.name}`: {collection.num_entities}")

    # free some memory
    del words, num_words
    if ran_batching:
        del text, num_text, model_output

    print(f"Creating indexes over vector embeddings using {metric_type} metric type and {index_type} index type.")
    collection.create_index(
        field_name="embedding",
        index_params={
            "metric_type": metric_type,
            "index_type": index_type,
            "params": params
        }
    )
    print("Finished creating indexes.")
    return


def insert_pca_embeddings_in_milvus(
    path_to_vocab,
    batch_size,
    pca_collection,
    embedding_collection,
    pca_model_path,
    metric_type="L2",
    index_type="IVF_FLAT",
    params={"nlist": 1024}
):
    """
    Insert vector embeddings into a defined Milvus collection

    Args:
        path_to_vocab (str): Path to dictionary text file.
        batch_size (int): The size of the batch sent to Triton for inference.
        model (InferenceServerClient): Defined TritonRemoteModel each word is sent to.
        pca_collection (pymilvus.Collection): Defined Milvus collection for each reduced embedding to be sent to.
        embedding_collection (pymilvus.Collection): Defined Milvus collection that the full dimensionaly embeddings come from.
        pca_model_path (str): Path to save the PCA model for later use. 
        metric_type (str): Metric used to measure similarity of vectors.
            For floating point vectors: "L2", "IP", "COSINE"
            For binary vectors: "JACCARD", "HAMMING"
        index_type (str): Type of index used to accelerate vector search.
            For floating point vectors: "FLAT", "IVF_FLAT", "IVF_SQ8", "IVF_PQ", "GPU_IVF_FLAT*", "GPU_IVF_PQ*>", "HNSW", "DISKANN*"
            For binary vectors: "BIN_FLAT", "BIN_IVF_FLAT"
        params (dict): See https://milvus.io/docs/index.md

    Returns:
        None
    """
    # read all words into list
    with open(path_to_vocab, "r") as file:
        words = file.read().splitlines()

    num_words = len(words)

    pca = IncrementalPCA(n_components=3, batch_size=batch_size)

    for batch in tqdm(
        range(0, num_words, batch_size),
        desc=f"Collecting full embeddings with batch size of {batch_size}",
        ncols=150,
        bar_format="{l_bar}{bar:10}{r_bar}"
    ):
        embeddings = []
        end_batch = batch + batch_size
        text = words[batch:end_batch]
        num_text = len(text)

        if num_text == batch_size:
            batch_range = list(range(batch, end_batch))
        else: # If last batch doesn't match the normal batch_size
            batch_range = list(range(batch, batch + num_text))

        response = embedding_collection.query(
            expr=f"id in {batch_range}",
            output_fields=["embedding"],
            consistency_level="Strong"
        )

        # Milvus returns the [id] and [embedding] in response[i].values()
        # but will sometimes switch the order, so I check to make sure
        # a list was grabbed (the embedding), otherwise grab the second index
        for i in range(num_text):
            temp = list(response[i].values())[0]
            if isinstance(temp, list):
                embeddings.append(temp)
            else:
                embeddings.append(list(response[i].values())[1])

        transformed_embeddings = pca.fit_transform(embeddings)

        pca_collection.insert([
            batch_range,
            text,
            transformed_embeddings
        ])

    joblib.dump(pca, pca_model_path)
    print(f"Fitted PCA model saved to: {pca_model_path}")

    pca_collection.flush()
    print(f"Number of entities in collection `{pca_collection.name}`: {pca_collection.num_entities}")

    # free some memory
    del words, num_words, pca, batch, embeddings, end_batch, text, num_text, batch_range, response, temp, transformed_embeddings

    print(f"Creating indexes over vector embeddings using {metric_type} metric type and {index_type} index type.")
    pca_collection.create_index(
        field_name="embedding",
        index_params={
            "metric_type": metric_type,
            "index_type": index_type,
            "params": params
        }
    )
    print("Finished creating indexes.")
    return