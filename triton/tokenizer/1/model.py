import json

import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer

class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args['model_config'])
        repository = model_config["parameters"]["REPOSITORY"]["string_value"]
        model_name = model_config["parameters"]["MODEL_NAME"]["string_value"]
        self.tokenizer = AutoTokenizer.from_pretrained(repository + "/" + model_name)

    def execute(self, requests):
        responses = []
        for request in requests:
            text = pb_utils.get_input_tensor_by_name(request, "INPUT_TEXT").as_numpy()[0].decode()
            encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')

            inference_response = pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor(key, np.array(encoded_input[key], dtype='int64'))
                for key in encoded_input
            ])

            responses.append(inference_response)              

        return responses
        # def mean_pooling(model_output, attention_mask):
        #     # copied from huggingface
        #     token_embeddings = model_output
        #     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        #     return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def finalize(self):

        print('Cleaning up tokenizer...')