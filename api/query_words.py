from app.embedding_utils import load_embeddings, load_word_dicts, count_populated, create_embedding, load_models
from app.query_utils import find_similar_words

import numpy as np
import code

###########################
### Script
###########################

cache_path = "res/word_embeddings_cache.npz"
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

tokenizer, model, device = load_models('sentence-transformers/all-MiniLM-L6-v2')
q = create_embedding("king", tokenizer, model, device) \
  - create_embedding("man", tokenizer, model, device) \
  + create_embedding("woman", tokenizer, model, device)
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