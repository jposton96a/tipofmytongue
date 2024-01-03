import sys

from dotenv import load_dotenv
from pymilvus import connections, Collection, MilvusException

from app.env_utils import EnvArgumentParser
from app.embedding_utils import create_embedding
from app.query_utils import k_similar_words
from app.triton_utils import TritonRemoteModel



def main(
    model_name,
    milvus_uri,
    triton_uri,
    connection_timeout=10
):
    # Milvus doesn't allow hyphens, so replace with underscores
    embedding_collection_name = model_name.replace("-", "_") if "-" in model_name else model_name
    pca_collection_name = embedding_collection_name + "_pca"

    # Establish connection to Milvus and Triton service
    try:
        connections.connect(alias="default", uri=milvus_uri, timeout=connection_timeout)
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
    load_dotenv()
    parser = EnvArgumentParser()
    parser.add_arg("MODEL_NAME", default="all-MiniLM-L6-v2", type=str)
    parser.add_arg("MILVUS_URI", default="grpc://localhost:19530", type=str)
    parser.add_arg("TRITON_URI", default="grpc://localhost:8001", type=str)
    args = parser.parse_args()

    main(
        args.MODEL_NAME,
        args.MILVUS_URI,
        args.TRITON_URI
    )