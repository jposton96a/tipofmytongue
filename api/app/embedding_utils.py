import numpy as np



def create_embedding(text, model):
    """
    Pass the text to the loaded Triton model for inference.

    Args:
        text (str): Text for inference.
        model (tritonclient.InferenceServerClient): Connection client for Triton model.

    Returns:
        embedding (np.array): Word embedding returned from the Triton request.
    """
    encoded_str = np.array([str.encode(text)])
    return model(1, encoded_str).numpy()