FROM nvcr.io/nvidia/tritonserver:23.10-py3
WORKDIR /src
RUN pip3 install transformers[torch]
CMD ["tritonserver", "--model-repository=/models", "--http-port", "8100", "--grpc-port", "8101", "--metrics-port", "8102"]