## Setting up Triton Inference Server
This Triton model repository is available at `s3://tipofmytongue-models-gpu/` and is public facing. This project requires an environment variable file at `triton/.env`.

```bash title="triton/.env"
AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID>
AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY>
AWS_DEFAULT_REGION=<AWS_DEFAULT_REGION>
MODEL_REPO="s3://tipofmytongue-models-gpu/all-MiniLM-L6-v2/"
```

There are currently two models available at the following locations:
* all-MiniLM-L6-v2: `s3://tipofmytongue-models-gpu/all-MiniLM-L6-v2/` (384 dimensions)
* gte-large: `s3://tipofmytongue-models-gpu/gte-large/` (1024 dimensions)

Triton can be started using Docker and the compose plugin. The service is located in the `docker-compose.yml` file named `triton`. 

To see each model repository, pull them down using the AWS cli:
```bash
aws s3 cp s3://tipofmytongue-models-gpu/ ./models/ --recursive
```

## Using Triton on GPU
Running Triton on a machine with a CUDA-enabled GPU will yeild the best results. If running on a GPU, make sure the following option exists:
1. Under the `triton` service in the `docker-compose.yml` file add the following:

    ```yaml title="docker-compose.yml"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ```

2. The `instance_group` option in each `config.pbtxt` file should be set to `KIND_GPU` as opposed to `KIND_CPU`. The count option is how many instances of the model you want to load. This is advantageous to handle two requests coming at the same time.

    ```yaml title="config.pbtxt"
    instance_group [
        {
            count: 2
            kind: KIND_GPU
        }
    ]
    ```
In this branch, these options should already be set.



## How Triton Works
Both of these models harness the "ensemble" model type. This allows you to create a pipeline to link models together. In our case, we link the "preprocess", "transformer", and "postprocess" models. The "preprocess" and "postprocess" models use the Triton python backend, allowing you to streamline these Python tasks within the Triton service. The "transformer" model is the actual embedding transformer model using the onnxruntime backend. This model was converted using the `optimum` [package](https://github.com/huggingface/optimum) that converts the model to ONNX format and quantizes it. This reduces the footprint and increases the efficieny of the model.

To run the Triton Inference Server, a model repository is required. A model repository has no limit to the number of models, also allowing multiple versions of each model. A typical model respository has the following structure:

```bash
model_repository/
├─ model_name/
│  ├─ 1/
│  │  ├─ model.plan
│  ├─ 2/
│  ├─ 3/
│  ├─ config.pbtxt
├─ preprocess/
│  ├─ 1/
│  │  ├─ model.py
│  ├─ config.pbtxt
├─ postprocess/
│  ├─ 1/
│  │  ├─ model.py
│  ├─ config.pbtxt
├─ ensemble/
│  ├─ 1/
│  ├─ config.pbtxt
├─ another_model/
.
.
.
```

with a typical `config.pbtxt` in the following form:

```text title="config.pbtxt"
name: "transformer"
platform: "onnxruntime_onnx"
max_batch_size : 0

input [
  {
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [ -1, -1 ]
  },
  {
    name: "attention_mask"
    data_type: TYPE_INT64
    dims: [ -1, -1 ]
  }
]

output [
  {
    name: "token_embeddings"
    data_type: TYPE_FP32
    dims: [ -1, -1, 384 ]
  }
]

model_warmup [
  {
    name : "transformer"
    batch_size: 1
    count: 2
    inputs {
      key: "input_ids"
      value: {
        data_type: TYPE_INT64
        dims: 1
        dims: 3
        input_data_file: "input_ids"
      }
    }
    inputs {
      key: "attention_mask"
      value: {
        data_type: TYPE_INT64
        dims: 1
        dims: 3
        input_data_file: "attention_mask"
      }
    }
  }
]

instance_group [
  {
    count: 2
    kind: KIND_GPU
  }
]

version_policy: { 
  latest: { num_versions: 1 }
}

response_cache { 
  enable: True 
}

optimization { execution_accelerators {
  cpu_execution_accelerator : [ {
    name : "openvino"
  }]
}}
```