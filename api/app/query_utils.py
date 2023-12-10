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



# Old functions
# def build_lookup_tables(dictionary, embeddings):
#     """
#     Build a 2-way lookup table for word embeddings.

#     Args:
#         lines (list): A list of words.
#         embeddings (list): A list of corresponding embeddings.

#     Returns:
#         lookup_word_embeddings: A dictionary mapping words to embeddings.
#         lookup_embedding_words: A dictionary mapping embeddings to words.
#     """
#     lookup_word_embeddings = {}
#     lookup_embedding_words = {}
#     for word, embedding in zip(dictionary, embeddings):
#         word = word.strip()
#         lookup_word_embeddings[word] = embedding
#         lookup_embedding_words[embedding.tobytes()] = word

#     return lookup_word_embeddings, lookup_embedding_words

# def find_similar_words(query_vector, embeddings, word_list, k):
#     """
#     Find similar words to a query vector using k-nearest neighbors algorithm.

#     Args:
#         query_vector (list): The vector to find similar words for.
#         embeddings (list): List of embedding vectors.
#         word_list (list): List of words associated with the embedding vectors.
#         k (int): Number of nearest neighbors to find.

#     Returns:
#         list: List of k similar words to the query vector.
#     """
#     from sklearn.neighbors import NearestNeighbors

#     # Create a NearestNeighbors object
#     nn = NearestNeighbors(n_neighbors=k)

#     # Fit the embeddings to the NearestNeighbors object
#     nn.fit(embeddings)

#     # Find the word_ids of the k nearest neighbors
#     distances, word_ids = nn.kneighbors([query_vector], k, return_distance=True)

#     # Get the corresponding words for the word_ids
#     # similar_words = [word_list[i] for i in word_ids[0]]
#     similar_words = [(word_list[id], id, distances[0][i]) for i, id in enumerate(word_ids[0])]

#     return similar_words

# def lookup_word_embedding(word_embeddings, query_word):
#     query = word_embeddings[query_word]
#     query = query / np.sqrt((query**2).sum())
#     return query

# def search_similar_embeddings(embeddings, lookup_embedding_words, query, count=10):
#     similarities = embeddings.dot(query)
#     sorted_ix = np.argsort(-similarities)

#     matches = []
#     for k in sorted_ix[:count]:
#       # Lookup word for the similar embedding
#       embedding_key = embeddings[k].tobytes()
#       word = None

#       # TODO: REMOVE - this is DUMB
#       # Can just use k to lookup the index in the dict...
#       if embedding_key in lookup_embedding_words:
#         word = lookup_embedding_words[embedding_key]
      
#       matches.append((word, k, similarities[k]))

#     return matches


# TODO - Remove global dependencies & extract these into query_utils package

# def similar_words(q, k=10):
#    return find_similar_words(q, embeddings, dictionary, k)

# def similar_svn(q, k=10, knn_count=100, c=0.1):
#     # Use KNN to find the nearest knn_count words using a basic distance function
#     # This helps narrow the search for the SVN, since training an SVN is expensive
#     knn_results = similar_words(q, knn_count)
#     local_embeddings = [embeddings[x[1]] for x in knn_results]


#     from sklearn import svm
#     # Append the query into the set of data to evaluate
#     x = np.concatenate([q[None,...], local_embeddings])
#     y = np.zeros(len(x))
#     y[0] = 1 # We have 1 positive sample

#     # Train the SVN
#     clf = svm.LinearSVC(class_weight='balanced', verbose=False, dual=True, max_iter=10000, tol=1e-6, C=c)
#     clf.fit(x, y)
    
#     # Only run inference on the knn result embeddings (skip the query included in X)
#     similarities = clf.decision_function(local_embeddings)
#     sorted_ix = np.argsort(-similarities)

#     # Build response format. Mirror the KNN result structure, but include similarity score instead of distance   
#     matches = []
#     for k in sorted_ix[:k]:
#         knn_result_mapping = knn_results[k]
#         matches.append((knn_result_mapping[0], knn_result_mapping[1], similarities[k]))
    
#     return matches