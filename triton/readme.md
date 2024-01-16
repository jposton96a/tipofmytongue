## Setting up Triton Inference Server
There are many different Triton model repositories available at `s3://tipofmytongue-models-gpu/` and are public facing. This project requires an environment variable file `.env`, in the root of this project, containing the model name, s3 bucket name, and AWS credentials.

```bash title=".env"
# triton/.env
MODEL_NAME=gte-small
MODEL_REPO=s3://tipofmytongue-models-gpu

AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID>
AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY>
AWS_DEFAULT_REGION=<AWS_DEFAULT_REGION>
```

The final path will concatenate the `MODEL_NAME` and `MODEL_REPO` with a forward slash (`/`).

There are currently six models available at the following locations:
* all-MiniLM-L6-v2: `s3://tipofmytongue-models-gpu/all-MiniLM-L6-v2/` (384 dimensions)
* all-distilroberta-v1: `s3://tipofmytongue-models-gpu/all-distilroberta-v1/` (768 dimensions)
* gte-large: `s3://tipofmytongue-models-gpu/gte-large/` (1024 dimensions)
* gte-base: `s3://tipofmytongue-models-gpu/gte-base/` (768 dimensions)
* gte-small: `s3://tipofmytongue-models-gpu/gte-small/` (384 dimensions)
* ember-v1: `s3://tipofmytongue-models-gpu/ember-v1/` (1024 dimensions)

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

## Setting up Triton Inference Server
There are many different Triton model repositories available at `s3://tipofmytongue-models/` and are public facing. This project requires an environment variable file at `triton/.env` containing AWS credentials and the model and s3 bucket path defined in the `docker-compose.yml` file.

```bash title="triton/.env"
# triton/.env
AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID>
AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY>
AWS_DEFAULT_REGION=<AWS_DEFAULT_REGION>
```

```bash title="docker-compose.yml"
# docker-compose.yml
services:
  triton:
    environment:
      MODEL: all-MiniLM-L6-v2
      MODEL_REPO: s3://tipofmytongue-models
```

The final path will concatenate these two variables with a forward slash (`/`).

There are currently six model repositories available at the following locations:
* all-MiniLM-L6-v2: `s3://tipofmytongue-models/all-MiniLM-L6-v2/` (384 dimensions)
* all-distilroberta-v1: `s3://tipofmytongue-models/all-distilroberta-v1/` (768 dimensions)
* gte-large: `s3://tipofmytongue-models/gte-large/` (1024 dimensions)
* gte-base: `s3://tipofmytongue-models/gte-base/` (768 dimensions)
* gte-small: `s3://tipofmytongue-models/gte-small/` (384 dimensions)
* ember-v1: `s3://tipofmytongue-models/ember-v1/` (1024 dimensions)

Triton can be started using Docker and the compose plugin. The service is located in the `docker-compose.yml` file named `triton`. 

To see each model repository, pull them down using the AWS cli:
```bash
aws s3 cp s3://tipofmytongue-models/ ./models/ --recursive
```

## Using Triton on GPU
Running Triton on a machine with a CUDA-enabled GPU will yeild the best results. If you are not running on a GPU, make sure the following option is removed:
1. Under the `triton` service in the `docker-compose.yml` file remove the following:

    ```yaml title="docker-compose.yml"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ```

2. The `instance_group` option in each `config.pbtxt` file should be set to `KIND_CPU` as opposed to `KIND_GPU`. The count option is how many instances of the model you want to load. This is advantageous to handle two requests coming at the same time.

    ```yaml title="config.pbtxt"
    instance_group [
        {
            count: 2
            kind: KIND_GPU
        }
    ]
    ```

The s3 bucket `s3://tipofmytongue-models/` will have these changes for CPU and the s3 bucket `s3://tipofmytongue-models-gpu/` will have these changes for GPU.


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
    kind: KIND_CPU
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