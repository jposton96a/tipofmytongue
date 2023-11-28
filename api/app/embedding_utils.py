import numpy as np
import openai
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


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
    return [entry.strip() for entry in lines]


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


def load_models(model_name):
    """
    Load models to specified device for future requests

    Args:
      - model_name: model name from Huggingface

    Returns:
      - tokenizer: pretrained tokenizer
      - model: pretrained model loaded to specified device
      - device: device to load model on (e.g. nvidia CUDA GPU)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    return tokenizer, model, device


def mean_pooling(model_output, attention_mask):
  token_embeddings = model_output[0]
  input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
  return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def create_embedding(text, tokenizer, model, device):
  encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)

  with torch.no_grad():
    model_output = model(**encoded_input)

  embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
  embeddings = torch.squeeze(F.normalize(embeddings, p=2, dim=1)).cpu()
  # response = openai.Embedding.create(model="text-embedding-ada-002", input=text)
  embedding = np.array(embeddings)
  # embedding = np.array(response["data"][0]["embedding"])
  norm_embedding = embedding / np.sqrt((embedding**2).sum())
  return norm_embedding