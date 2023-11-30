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

    def finalize(self):

        print('Cleaning up tokenizer...')