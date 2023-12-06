import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from app.embedding_utils import create_embedding, load_embeddings, load_word_dicts, count_populated, TritonRemoteModel
from app.transform_utils import create_or_load_transform


# cache_path = "res/chatgpt_embedding_subset_100.npz"
cache_path = "res/word_embeddings_cache.npz"
dict_path = "res/words.txt"

transform_model_path = "res/pca_transform.pkl"
transformed_embeddings_path = "res/pca_transformed_embeddings.npy"

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

transform_model, reduced_embeddings = create_or_load_transform(
    embeddings=embeddings, 
    transform_model_path=transform_model_path, 
    transformed_embeddings_path=transformed_embeddings_path
)

model = TritonRemoteModel("http://localhost:8100", "gte-large")
input_vector = create_embedding("king", model)
transformed_search_vector = transform_model.transform(input_vector.reshape(1, -1))

print (transformed_search_vector)

# Create a 3D scatter plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
x, y, z = reduced_embeddings[:, 0], reduced_embeddings[:, 1], reduced_embeddings[:, 2]
ax.scatter(x, y, z)

# Plot the search vector
ax.scatter(transformed_search_vector[0][0], transformed_search_vector[0][1], transformed_search_vector[0][2], color='red', label='Search Vector')


# Label each point with its corresponding word
for i, word in enumerate(dictionary[:embeddings_count]):
    ax.text(x[i], y[i], z[i], word, size=10, zorder=1, color='k')

ax.set_title('3D PCA of Word Embeddings')
ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
ax.set_zlabel('PCA3')
# plt.show()
plt.savefig('plot.png')