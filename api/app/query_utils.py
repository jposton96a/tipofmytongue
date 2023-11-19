import numpy as np

def build_lookup_tables(dictionary, embeddings):
    """
    Build a 2-way lookup table for word embeddings.

    Args:
        lines (list): A list of words.
        embeddings (list): A list of corresponding embeddings.

    Returns:
        lookup_word_embeddings: A dictionary mapping words to embeddings.
        lookup_embedding_words: A dictionary mapping embeddings to words.
    """
    lookup_word_embeddings = {}
    lookup_embedding_words = {}
    for word, embedding in zip(dictionary, embeddings):
        word = word.strip()
        lookup_word_embeddings[word] = embedding
        lookup_embedding_words[embedding.tobytes()] = word

    return lookup_word_embeddings, lookup_embedding_words

def find_similar_words(query_vector, embeddings, word_list, k):
    """
    Find similar words to a query vector using k-nearest neighbors algorithm.

    Args:
        query_vector (list): The vector to find similar words for.
        embeddings (list): List of embedding vectors.
        word_list (list): List of words associated with the embedding vectors.
        k (int): Number of nearest neighbors to find.

    Returns:
        list: List of k similar words to the query vector.
    """
    from sklearn.neighbors import NearestNeighbors

    # Create a NearestNeighbors object
    nn = NearestNeighbors(n_neighbors=k)

    # Fit the embeddings to the NearestNeighbors object
    nn.fit(embeddings)

    # Find the word_ids of the k nearest neighbors
    distances, word_ids = nn.kneighbors([query_vector], k, return_distance=True)

    # Get the corresponding words for the word_ids
    # similar_words = [word_list[i] for i in word_ids[0]]
    similar_words = [(word_list[id], id, distances[0][i]) for i, id in enumerate(word_ids[0])]

    return similar_words

def lookup_word_embedding(word_embeddings, query_word):
    query = word_embeddings[query_word]
    query = query / np.sqrt((query**2).sum())
    return query

def search_similar_embeddings(embeddings, lookup_embedding_words, query, count=10):
    similarities = embeddings.dot(query)
    sorted_ix = np.argsort(-similarities)

    matches = []
    for k in sorted_ix[:count]:
      # Lookup word for the similar embedding
      embedding_key = embeddings[k].tobytes()
      word = None

      # TODO: REMOVE - this is DUMB
      # Can just use k to lookup the index in the dict...
      if embedding_key in lookup_embedding_words:
        word = lookup_embedding_words[embedding_key]
      
      matches.append((word, k, similarities[k]))

    return matches
