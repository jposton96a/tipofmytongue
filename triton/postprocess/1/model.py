import torch
import torch.nn.functional as F
import triton_python_backend_utils as pb_utils


def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    pool = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    return F.normalize(pool, p=2, dim=1)


class TritonPythonModel:
    def initialize(self, args):
        pass

    def execute(self, requests):
        responses = []
        for request in requests:
            last_hidden_state = pb_utils.get_input_tensor_by_name(request, "token_embeddings").as_numpy()
            attention_mask = pb_utils.get_input_tensor_by_name(request, "attention_mask").as_numpy()

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[pb_utils.Tensor(
                        "embedding",
                        average_pool(
                            torch.from_numpy(last_hidden_state),
                            torch.from_numpy(attention_mask)
                        ).numpy()
                    )]
                )
            )              

        return responses

    def finalize(self):

        print('Cleaning up tokenizer...')