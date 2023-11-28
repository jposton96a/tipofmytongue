import os

import joblib
import numpy as np
from sklearn.decomposition import PCA


def create_or_load_transform(embeddings, transformed_embeddings_path, transform_model_path):
    pca = None
    transformed_embeddings = None

    if os.path.isfile(transformed_embeddings_path) or os.path.isfile(transform_model_path):
        print("Detected existing PCA model...", end='')
        pca = joblib.load(transform_model_path)
        transformed_embeddings = np.load(transformed_embeddings_path)
        print("loaded!")

    else:
        # Perform PCA to reduce the embeddings to 3 dimensions
        print("No existing model found. Fitting new transform model")
        pca = PCA(n_components=3)
        transformed_embeddings = pca.fit_transform(embeddings)

        print("Saving...", end="")
        save_model(pca, transformed_embeddings, transform_model_path, transformed_embeddings_path)
        print ("done!")
    
    if pca is None or transformed_embeddings is None:
        raise Exception("Failed to setup transform model")
    
    return [pca, transformed_embeddings]


def save_model(model, transformed_embeddings, transform_model_path, transformed_embeddings_path):
    np.save(transformed_embeddings_path, transformed_embeddings)
    joblib.dump(model, transform_model_path)