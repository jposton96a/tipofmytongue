import boto3
from os import path

BUCKET_NAME = "tipofmytongue"
BUCKET_CACHE_DIR = ""

EMBEDDING_CACHE_DIR="res/"

# Filename to pull (assumes the destination & source have the same basename)
def download_embeddings(cache_filename='word_embeddings_cache.npz.chk_non_norm_466503'):
    s3 = boto3.client('s3')
    target_path = path.join(EMBEDDING_CACHE_DIR, cache_filename)
    
    print(f"Downloading {cache_filename} to {target_path} ...", end="")
    s3.download_file(
        Bucket=BUCKET_NAME, 
        Key=path.join(BUCKET_CACHE_DIR, cache_filename), 
        Filename=target_path
    )
    print("done!")


if __name__ == "__main__":
    download_embeddings(cache_filename="requirements.txt")