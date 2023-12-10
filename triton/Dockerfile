FROM nvcr.io/nvidia/tritonserver:23.10-py3
WORKDIR /src

RUN pip3 install transformers[torch]

CMD ["tritonserver", "--model-repository=/models"]