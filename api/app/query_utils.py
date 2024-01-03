import numpy as np



def k_similar_words(embedding, embedding_collection, pca_collection, k=10):
    """
    Queries the Milvus collection for the closest word embedding provided.
    The search uses the pre-indexed metric_type=L2 and index_type IVF_FLAT.

    Args:
        embedding (np.array): Word embedding to compare against.
        embedding_collection (pymilvus.Collection): Milvus collection for search.
        pca_collection (pymilvus.Collection): Milvus collection for query.

    Returns:
        (tuple): Words, embeddings, and ids that are most similar to the provided embedding.
    """
    response = embedding_collection.search(
        data=embedding,
        anns_field="embedding",
        param={
            "metric_type": "L2",
            "params": {"nprobe": 10},
        },
        limit=k,
        output_fields=["word"]
    )[0]

    words = [res.to_dict()["entity"]["word"] for res in response]

    response = pca_collection.query(
        expr=f"word in {words}",
        output_fields=["embedding", "id"]
    )

    # Milvus return a python list of np.float32, so we convert the list
    # to a np.array and back to a list to make all the values native floats
    embeddings = [np.array(res["embedding"]).tolist() for res in response]
    ids = [res["id"] for res in response]

    return [(words[i], embeddings[i], ids[i]) for i in range(k)]


def query_pca_word_embedding(ids, pca_collection):
    """
    A simple query to grab the 3-dimensional word embedding for the given word.
    We query the ids because Milvus return the query in order of ids, therefore it does not
    preserve the order we feed in. Because of this we have to sort the ids and then index the results.
    The Milvus query returns a python list with np.float32 (e.g. [np.float32(), np.float32(), ...]).
    To convert the np.float32 to the native python float we pass it into np.array(list).tolist()

    Args:
        ids (list): List of word ids to be queried on.
        pca_collection (pymilvus.Collection): Milvus collection for query.

    Returns:
        (list): List of the 3-dimensional pca word embeddings.
    """
    indexes = [i[0] for i in sorted(enumerate(ids), key=lambda x:x[1])]
    query = pca_collection.query(
        expr=f"id in {ids}",
        output_fields=["embedding"]
    )
    return [np.array(x['embedding']).tolist() for _, x in sorted(zip(indexes, query))]