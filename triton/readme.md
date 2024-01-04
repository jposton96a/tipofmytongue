triton/
├─ trt_models/
│  ├─ model_name/
│  │  ├─ 1/
│  │  │  ├─ model.plan
│  │  ├─ 2/
│  │  ├─ 3/
│  │  ├─ config.pbtxt
│  ├─ preprocess/
│  │  ├─ 1/
│  │  │  ├─ model.py
│  │  ├─ config.pbtxt
│  ├─ postprocess/
│  │  ├─ 1/
│  │  │  ├─ model.py
│  │  ├─ config.pbtxt
│  ├─ ensemble/
│  │  ├─ 1/
│  │  ├─ config.pbtxt


## Using Triton on CPU
If you're starting up all the services you will need to make two changes:
1. In `docker-compose.yml` remove the deploy key and everything under it.
2. In every `config.pbtxt` under the triton model repository change every instance_group `KIND_GPU` to `KIND_CPU`.

If you know a better way to make these steps easuer let me know.

Also download the ONNX transformer model for `all-MiniLM-L6-v2` from [here](https://huggingface.co/optimum/all-MiniLM-L6-v2/tree/main) and place at `triton/transformer/1/model.onnx`.

To rebuild the embeddings cache you will need to launch `triton-server` using `docker compose up -d triton-server` but expose port 8100 as well in the docker-compose file.

As of 12/1:
Added new model "gte-large" from Huggingface.
The model was 1.3GB but after quantization we can get it down to ~300MB. Has 1024 dimensions as opposed to 384 from "all-MiniLM-L6-v2".
I also added a postprocess step to triton that does mean pooling and normalization.