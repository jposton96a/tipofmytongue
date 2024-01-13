# Generated using the compose tool at https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/compose.md
# Use the command:
#
# python compose.py \
# --backend pytorch \
# --container-version 23.12 \
# --image min,nvcr.io/nvidia/tritonserver:23.12-py3-min \
# --image full,nvcr.io/nvidia/tritonserver:23.12-pyt-python-py3 \
# --enable-gpu

#
# Multistage build.
#
ARG TRITON_VERSION=2.41.0
ARG TRITON_CONTAINER_VERSION=23.12

FROM nvcr.io/nvidia/tritonserver:23.12-pyt-python-py3 AS full
FROM nvcr.io/nvidia/tritonserver:23.12-py3-min

ARG TRITON_VERSION
ARG TRITON_CONTAINER_VERSION

ENV TRITON_SERVER_VERSION=${TRITON_VERSION}
ENV NVIDIA_TRITON_SERVER_VERSION=${TRITON_CONTAINER_VERSION}
LABEL com.nvidia.tritonserver.version="${TRITON_SERVER_VERSION}"

ENV PATH=/opt/tritonserver/bin:${PATH}
ENV UCX_MEM_EVENTS=no
ENV LD_LIBRARY_PATH=/opt/hpcx/ucc/lib/:/opt/hpcx/ucx/lib/:${LD_LIBRARY_PATH}
ENV TF_ADJUST_HUE_FUSED=1
ENV TF_ADJUST_SATURATION_FUSED=1
ENV TF_ENABLE_WINOGRAD_NONFUSED=1
ENV TF_AUTOTUNE_THRESHOLD=2
ENV TRITON_SERVER_GPU_ENABLED=1

# Create a user that can be used to run triton as
# non-root. Make sure that this user to given ID 1000. All server
# artifacts copied below are assign to this user.
ENV TRITON_SERVER_USER=triton-server
ENV DEBIAN_FRONTEND=noninteractive
ENV TCMALLOC_RELEASE_RATE=200
ENV DCGM_VERSION=3.2.6

# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive
ENV TCMALLOC_RELEASE_RATE=200
ENV DCGM_VERSION=3.2.6

# Common dependencies. FIXME (can any of these be conditional? For
# example libcurl only needed for GCS?)
# Install boost version >= 1.78 for boost::span
# Current libboost-dev apt packages are < 1.78, so install from tar.gz
# Set TCMALLOC_RELEASE_RATE for users setting LD_PRELOAD with tcmalloc
# Install DCGM. Steps from https://developer.nvidia.com/dcgm#Downloads
RUN userdel tensorrt-server > /dev/null 2>&1 || true \
    && if ! id -u $TRITON_SERVER_USER > /dev/null 2>&1 ; then \
    useradd $TRITON_SERVER_USER; fi \
    && [ `id -u $TRITON_SERVER_USER` -eq 1000 ] \
    && [ `id -g $TRITON_SERVER_USER` -eq 1000 ] \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    clang curl dirmngr git gperf \
    libb64-0d libcurl4-openssl-dev \
    libgoogle-perftools-dev libjemalloc-dev \
    libnuma-dev libre2-9 software-properties-common \
    wget libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && wget -O /tmp/boost.tar.gz https://archives.boost.io/release/1.80.0/source/boost_1_80_0.tar.gz \
    && (cd /tmp && tar xzf boost.tar.gz) \
    && cd /tmp/boost_1_80_0 \
    && ./bootstrap.sh --prefix=/usr \
    && ./b2 install \
    && rm -rf /tmp/boost* \
    && curl -o /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb \
    && apt install /tmp/cuda-keyring.deb \
    && rm /tmp/cuda-keyring.deb \
    && apt-get update \
    && apt-get install -y datacenter-gpu-manager=1:3.2.6 \
    && ln -sf ${_CUDA_COMPAT_PATH}/lib.real ${_CUDA_COMPAT_PATH}/lib \
    && echo ${_CUDA_COMPAT_PATH}/lib > /etc/ld.so.conf.d/00-cuda-compat.conf \
    && ldconfig && rm -f ${_CUDA_COMPAT_PATH}/lib

WORKDIR /opt/tritonserver
RUN rm -fr /opt/tritonserver/*
ENV NVIDIA_PRODUCT_NAME="Triton Server"
COPY docker/entrypoint.d/ /opt/nvidia/entrypoint.d/

ENV NVIDIA_BUILD_ID=77457706
LABEL com.nvidia.build.id=77457706
LABEL com.nvidia.build.ref=133242c14ca49b3fbb65686b2403294d36ddc21c

WORKDIR /opt/tritonserver
COPY --chown=1000:1000 --from=full /opt/tritonserver/LICENSE .
COPY --chown=1000:1000 --from=full /opt/tritonserver/TRITON_VERSION .
COPY --chown=1000:1000 --from=full /opt/tritonserver/bin bin/
COPY --chown=1000:1000 --from=full /opt/tritonserver/lib lib/
COPY --chown=1000:1000 --from=full /opt/tritonserver/include include/
# Copying over backends 
COPY --chown=1000:1000 --from=full /opt/tritonserver/backends/pytorch /opt/tritonserver/backends/pytorch

# Top-level /opt/tritonserver/backends not copied so need to explicitly set permissions here
RUN chown triton-server:triton-server /opt/tritonserver/backends
#  Copying over repoagents 
#  Copying over caches 

LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true
COPY --chown=1000:1000 --from=full /usr/bin/serve /usr/bin/.

RUN apt-get install -y --no-install-recommends python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir transformers[torch]

CMD ["sh", "-c", "tritonserver --model-repository=${MODEL_REPO}"]