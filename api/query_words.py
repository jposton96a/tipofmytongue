import sys

from pymilvus import connections, Collection, MilvusException

from app.embedding_utils import create_embedding
from app.query_utils import k_similar_words
from app.triton_utils import TritonRemoteModel



def main(
    model_name,
    milvus_uri,
    triton_uri
):
    # Milvus doesn't allow hyphens, so replace with underscores
    embedding_collection_name = model_name.replace("-", "_") if "-" in model_name else model_name
    pca_collection_name = embedding_collection_name + "_pca"

    # Establish connection to Milvus and Triton service
    try:
        connections.connect(alias="default", uri=milvus_uri)
        model = TritonRemoteModel(uri=triton_uri, model_name=model_name)
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

    q = create_embedding("king", model) - create_embedding("man", model) + create_embedding("woman", model)
    print(k_similar_words(q, embedding_collection, pca_collection))



if __name__ == "__main__":
    main(
        model_name="gte-small",
        milvus_uri="grpc://localhost:19530",
        triton_uri="grpc://localhost:8001"
    )