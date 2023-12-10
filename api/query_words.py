import sys
import code
from pymilvus import connections, Collection, MilvusException

from app.embedding_utils import create_embedding
from app.query_utils import k_similar_words
from app.triton_utils import TritonRemoteModel


###########################
### App Dependencies
###########################

transform_model_path = "res/pca_transform.pkl"
embedding_collection_name = "tipofmytongue"
pca_collection_name = "tipofmytongue_pca"

milvus_uri = "grpc://localhost:19530"
triton_uri = "grpc://localhost:8001"
model_name = "gte-large"
connection_timeout = 10

# Establish connection to Milvus and Triton service
try:
    connections.connect(alias="default", uri=milvus_uri, timeout=connection_timeout)
    model = TritonRemoteModel(url=triton_uri, model_name=model_name)
except MilvusException as e:
    print(f"Could not establish connection to Milvus: {e}")
    sys.exit(0)
except ConnectionRefusedError as e:
    print(f"Could not establish connection to Triton: {e}")
    sys.exit(0)

embedding_collection = Collection(embedding_collection_name)
embedding_collection.load()

pca_collection = Collection(pca_collection_name)
pca_collection.load()



###########################
### Script
###########################

q = create_embedding("king", model) - create_embedding("man", model) + create_embedding("woman", model)
print(k_similar_words(q, embedding_collection, pca_collection))



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