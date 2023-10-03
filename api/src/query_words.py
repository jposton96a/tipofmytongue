import code
import numpy as np
import openai

###########################
### FUNCTIONS
###########################

def load_embeddings(file_path):
    """
    Load the embeddings from the numpy array file.

    Args:
    - file_path (str): The path to the numpy array file.

    Returns:
    - embeddings (numpy.ndarray): The loaded embeddings.
    """
    print("Loading embeddings")
    data = np.load(file_path)
    embeddings = data["embeddings"]
    return embeddings


def load_word_dicts(file_path):
    """
    Load the words from the file.

    Args:
    - file_path (str): The path to the file containing the words.

    Returns:
    - lines (list): The lines read from the file.
    """
    print("Loading word dictionary")
    file = open(file_path, "r")
    lines = file.readlines()
    file.close()
    del file
    return lines


def count_populated(a: list[np.ndarray], prefix: bool = True):
    """
    Count the populated entries in a set of embeddings

    Args:
      - a: the input array
      - prefix: a boolean flag indicating whether to assume all populated elements are at the front

    Returns:
        _type_: _description_
    """
    count_empty = 0
    for i, line in enumerate(a):
      if line.nonzero()[0].size == 0 or np.any(np.isnan(line)):
        # Count every time we encounter an empty cell
        count_empty = count_empty + 1

        # `prefix`=True:
        # Assumes all the populated elements are at the front, and
        # anything after an empty index will also be empty
        if prefix:
          return i

    # Return the final count
    return len(a) - count_empty

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

def create_embedding(text):
    response = openai.Embedding.create(model="text-embedding-ada-002", input=text)
    embedding = np.array(response["data"][0]["embedding"])
    norm_embedding = embedding / np.sqrt((embedding**2).sum())
    return norm_embedding

###########################
### Script
###########################

cache_path = "res/word_embeddings_cache.npz.chk_non_norm_466503"
dict_path = "res/words.txt"

embeddings = load_embeddings(cache_path)
dictionary = load_word_dicts(dict_path)
print(f"Loaded {len(dictionary)} words from {dict_path}")

# The length of the embeddings will always match the dictionary.
# Some or all of the indexes may be populated
embeddings_count = count_populated(embeddings)
print(f"Loaded {embeddings_count} embeddings from {cache_path}")
if embeddings_count == 0:
    print("Cache empty! Exiting")
    exit(-1)

# Build lookup tables for finding an embedding for a given word, or looking up a word
# using a similar embedding
# NEEDED FOR EVERYTHING ELSE BELOW
# word_embeddings, embedding_words = build_lookup_tables(dictionary, embeddings)

# Simple wrapper for querying similar words
def similar_words(q, k=10):
   return find_similar_words(q, embeddings, dictionary, k)

def similar_svn(q, k=10, knn_count=100, c=0.1):
    # Use KNN to find the nearest knn_count words using a basic distance function
    # This helps narrow the search for the SVN, since training an SVN is expensive
    knn_results = similar_words(q, knn_count)
    local_embeddings = [embeddings[x[1]] for x in knn_results]


    from sklearn import svm
    # Append the query into the set of data to evaluate
    x = np.concatenate([q[None,...], local_embeddings])
    y = np.zeros(len(x))
    y[0] = 1 # We have 1 positive sample

    # Train the SVN
    clf = svm.LinearSVC(class_weight='balanced', verbose=False, dual=True, max_iter=10000, tol=1e-6, C=c)
    clf.fit(x, y)
    
    # Only run inference on the knn result embeddings (skip the query included in X)
    similarities = clf.decision_function(local_embeddings)
    sorted_ix = np.argsort(-similarities)

    # Build response format. Mirror the KNN result structure, but include similarity score instead of distance   
    matches = []
    for k in sorted_ix[:k]:
        knn_result_mapping = knn_results[k]
        matches.append((knn_result_mapping[0], knn_result_mapping[1], similarities[k]))
    
    return matches

q = create_embedding("king") - create_embedding("man") + create_embedding("woman")
print(similar_svn(q))
# Drop into Python Shell
# python -i foo.py
code.interact(local=locals())

# Sample Interpreter 
# find_similar_words(q, embeddings, dictionary, k=10)


###########################
### Scratch
###########################

### BELOW - Comparison of different query styles 
# (2 variations of knn + their normalized variants)

# Query Examples
# query_word = "zillion"
# query = lookup_word_embedding(word_embeddings, query_word)


# print("Matches1")
# matches = search_similar_embeddings(embeddings, embedding_words, query, count=8)
# print(matches)

# print("Matches2")
# matches2 = find_similar_words(query, embeddings, dictionary, k=10) ## GPT gave me this one lol
# print(matches2)


# ### Normalize - NOTE::: THIS KILLS THE RAM

# print("Normalizing Embedding cache...")
# del(word_embeddings)
# del(embedding_words)
# norm_embeddings = embeddings / np.sqrt((embeddings**2).sum(1, keepdims=True)) # L2 normalize
# norm_word_embeddings, norm_embedding_words = build_lookup_tables(dictionary, norm_embeddings)
# query = convert_to_query_embedding(norm_word_embeddings, query_word)

# print("Matches - Normalized")
# matches3 = search_similar_embeddings(norm_embeddings, norm_embedding_words, query, count=8)
# print(matches3)

# print("Matches 2 - Normalized")
# matches4 = find_similar_words(query, embeddings, dictionary, k=10) ## GPT gave me this one lol
# print(matches4)