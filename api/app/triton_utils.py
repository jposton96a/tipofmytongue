"""
requirements for model quantization:
    pip install optimum[onnxruntime]
    pip install tritonclient[all]
"""
import torch
import typing
import numpy as np
from urllib.parse import urlparse



class TritonRemoteModel:
    def __init__(self, uri: str, model_name: str):
        parsed_url = urlparse(uri)
        if parsed_url.scheme == "http":
            from tritonclient.http import InferenceServerClient, InferInput

            self.client = InferenceServerClient(parsed_url.netloc)
            self.model_name = model_name
            self.metadata = self.client.get_model_metadata(self.model_name)

            def create_input_placeholders(batch_size) -> typing.List[InferInput]:
                return [
                    InferInput(i['name'], [batch_size], i['datatype']) for i in self.metadata['inputs']
                ]

        else:
            from tritonclient.grpc import InferenceServerClient, InferInput

            self.client = InferenceServerClient(parsed_url.netloc)
            self.model_name = model_name
            self.metadata = self.client.get_model_metadata(self.model_name, as_json=True)

            def create_input_placeholders(batch_size) -> typing.List[InferInput]:
                return [
                    InferInput(i['name'], [batch_size], i['datatype']) for i in self.metadata['inputs']
                ]

        self._create_input_placeholders_fn = create_input_placeholders
    
    @property
    def runtime(self):
        return self.metadata.get("backend", self.metadata.get("platform"))

    def __call__(self, batch_size=1, *args, **kwargs) -> typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, ...]]:
        inputs = self._create_inputs(batch_size, *args, **kwargs)
        response = self.client.infer(model_name=self.model_name, inputs=inputs)
        result = []
        for output in self.metadata['outputs']:
            tensor = torch.tensor(response.as_numpy(output['name']))
            result.append(tensor)
        return result[0] if len(result) == 1 else result

    def _create_inputs(self, batch_size, *args, **kwargs):
        args_len, kwargs_len = len(args), len(kwargs)
        if not args_len and not kwargs_len:
            raise RuntimeError("No inputs provided.")
        if args_len and kwargs_len:
            raise RuntimeError("Cannot specify args and kwargs at the same time")

        placeholders = self._create_input_placeholders_fn(batch_size)

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


def convert_model_to_onnx(model_id, save_path=None):
    from optimum.onnxruntime import ORTModelForFeatureExtraction

    model = ORTModelForFeatureExtraction.from_pretrained(model_id, from_transformers=True)
    if save_path:
        model.save_pretrained(save_path)

    return model


def quantize_model(model_id, save_path):
    # repo: https://github.com/philschmid/optimum-transformers-optimizations/tree/master
    # made by an official HF employee
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    # create ORTQuantizer and define quantization configuration
    model = convert_model_to_onnx(model_id)
    dynamic_quantizer = ORTQuantizer.from_pretrained(model)
    dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)

    # apply the quantization configuration to the model
    model_quantized_path = dynamic_quantizer.quantize(
        save_dir=save_path,
        quantization_config=dqconfig
    )
    return model_quantized_path


def create_warmup_file(save_path, np_array=None, string=False, img_path=None):
    # For Triton model warmup, typical model input is a np.array, string, or image
    # The warmup files must be in bytes, so use this function to create them.
    if hasattr(np_array, 'shape'):
        np_array.tofile(save_path)
    elif isinstance(img_path, str):
        import cv2
        img = cv2.imread(img_path)
        img.tofile(save_path)
    elif isinstance(string, str):
        from tritonclient.utils import serialize_byte_tensor
        serialized = serialize_byte_tensor(np.array([string.encode("utf-8")], dtype=object))
        with open(save_path, "wb") as f:
            f.write(serialized.item())
    else:
        print("Invalid input. Input a numpy array, string, or image path")


# if __name__ == "__main__":
    ##################################################
    # # Model: thenlper/gte-large
    ##################################################
    # quantize_model("thenlper/gte-large", "./model")

    # # tokenizer
    # create_warmup_file(
    #     "../../triton/gte-large/tokenizer/warmup/input_text",
    #     string="king"
    # )

    # # transformer
    # create_warmup_file(
    #     "../../triton/gte-large/transformer/warmup/input_ids",
    #     np.array([[101, 2332, 102]])
    # )
    # create_warmup_file(
    #     "../../triton/gte-large/transformer/warmup/attention_mask",
    #     np_array=np.array([[1, 1, 1]])
    # )

    # # postprocess
    # create_warmup_file(
    #     "../../triton/gte-large/postprocess/warmup/token_embeddings",
    #     np_array=np.random.normal(0, 0.1, size=(1, 3, 1024))
    # )
    # create_warmup_file(
    #     "../../triton/gte-large/postprocess/warmup/attention_mask",
    #     np_array=np.array([[1, 1, 1]])
    # )


    ##################################################
    # # Model: sentence-transformers/all-MiniLM-L6-v2
    ##################################################
    # quantize_model("sentence-transformers/all-MiniLM-L6-v2", "./model")

    # # tokenizer
    # create_warmup_file(
    #     "../../triton/all-MiniLM-L6-v2/tokenizer/warmup/input_text",
    #     string="king"
    # )

    # # transformer
    # create_warmup_file(
    #     "../../triton/all-MiniLM-L6-v2/transformer/warmup/input_ids",
    #     np.array([[101, 2332, 102]])
    # )
    # create_warmup_file(
    #     "../../triton/all-MiniLM-L6-v2/transformer/warmup/attention_mask",
    #     np_array=np.array([[1, 1, 1]])
    # )

    # # postprocess
    # create_warmup_file(
    #     "../../triton/all-MiniLM-L6-v2/postprocess/warmup/token_embeddings",
    #     np_array=np.random.normal(0, 0.1, size=(1, 3, 384))
    # )
    # create_warmup_file(
    #     "../../triton/all-MiniLM-L6-v2/postprocess/warmup/attention_mask",
    #     np_array=np.array([[1, 1, 1]])
    # )