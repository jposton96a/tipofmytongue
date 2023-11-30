import typing

import torch
import numpy as np
import tritonclient.http as httpclient

from transformers import AutoTokenizer
from urllib.parse import urlparse


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


# def mean_pooling(model_output, attention_mask):
#     # copied from huggingface
#     token_embeddings = model_output
#     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#     return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def create_embedding(text, model):
    model_output = model(np.array([str.encode(text)]))
    embedding = np.array(model_output)
    # norm_embedding = embedding / np.sqrt((embedding**2).sum())
    return embedding


class TritonRemoteModel:
    def __init__(self, url: str, model: str):
        parsed_url = urlparse(url)
        if parsed_url.scheme == "http":
            self.client = httpclient.InferenceServerClient(parsed_url.netloc)
            self.model_name = model
            self.metadata = self.client.get_model_metadata(self.model_name)
    
    @property
    def runtime(self):
        return self.metadata.get("backend", self.metadata.get("platform"))

    def __call__(self, *args, **kwargs) -> typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, ...]]:
        inputs = self._create_inputs(*args, **kwargs)
        response = self.client.infer(model_name=self.model_name, inputs=inputs)
        result = []
        for output in self.metadata['outputs']:
            tensor = torch.as_tensor(response.as_numpy(output['name']))
            result.append(tensor)
        return result[0][0] if len(result) == 1 else result

    def _create_inputs(self, *args, **kwargs):
        args_len, kwargs_len = len(args), len(kwargs)
        if not args_len and not kwargs_len:
            raise RuntimeError("No inputs provided.")
        if args_len and kwargs_len:
            raise RuntimeError("Cannot specify args and kwargs at the same time")
        
        placeholders = [
            httpclient.InferInput(i['name'], [int(s) for s in args[index].shape], i['datatype']) for index, i in enumerate(self.metadata['inputs'])
        ]
        if args_len:
            if args_len != len(placeholders):
                raise RuntimeError(f"Expected {len(placeholders)} inputs, got {args_len}.")
            for input, value in zip(placeholders, args):
                input.set_data_from_numpy(value)
        else:
            for input in placeholders:
                value = kwargs[input.name]
                input.set_data_from_numpy(value)
        return placeholders