FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 AS phantora-pytorch

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
        ccache curl openssh-server pdsh \
        libpng-dev libjpeg-dev git \
        libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
        libncursesw5-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir /var/run/sshd
RUN /usr/sbin/update-ccache-symlinks
RUN mkdir -p /var/cache/ccache && \
    ccache --set-config=cache_dir=/var/cache/ccache

WORKDIR /phantora
RUN git clone https://github.com/pyenv/pyenv /phantora/pyenv
WORKDIR /phantora/pyenv/plugins/python-build
RUN ./install.sh
RUN mkdir /usr/local/python3.11.9
RUN python-build --verbose 3.11.9 /usr/local/python3.11.9
ENV PATH=/usr/local/python3.11.9/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/python3.11.9/lib

COPY pytorch /phantora/pytorch
WORKDIR /phantora/pytorch
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV TORCH_CUDA_ARCH_LIST="8.0;9.0"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV PYTORCH_BUILD_VERSION=2.7.1
ENV PYTORCH_BUILD_NUMBER=1
ENV USE_CUDNN=0
ENV USE_CUSPARSELT=0
# ENV _GLIBCXX_USE_CXX11_ABI=1

ARG MAX_JOBS

RUN python3 -m pip install --no-cache-dir -r requirements.txt
# may take a while
RUN --mount=type=cache,target=/var/cache/ccache \
    MAX_JOBS=${MAX_JOBS:-$(( $(nproc) / 4 ))} \
    python3 -m pip install --no-cache-dir --verbose -e .
# may also take a while
RUN FLASH_ATTN_CUDA_ARCHS="80;90" \
    MAX_JOBS=${MAX_JOBS:-$(( $(nproc) / 4 ))} \
    python3 -m pip install --no-cache-dir --verbose --no-build-isolation flash-attn==2.7.3

WORKDIR /phantora
# torchtitan dependencies somehow need to be installed manually
RUN curl -Lo torchtitan-requirements.txt https://raw.githubusercontent.com/pytorch/torchtitan/refs/tags/v0.1.0/.ci/docker/requirements.txt && \
    python3 -m pip install --no-cache-dir -r torchtitan-requirements.txt
RUN python3 -m pip install --no-cache-dir megatron-core==0.13.1 transformers==4.41.2 deepspeed==0.17.5 torchtitan==0.1.0

# DeepSpeed needs passwordless ssh
COPY config/sshconfig /root/.ssh/config
COPY config/id_ed25519 /root/.ssh/id_ed25519
COPY config/id_ed25519.pub /root/.ssh/id_ed25519.pub
COPY config/id_ed25519.pub /root/.ssh/authorized_keys
RUN chmod 600 /root/.ssh/id_ed25519


FROM phantora-pytorch AS phantora-local

WORKDIR /phantora
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
      libopenmpi-dev vim iproute2 && \
    rm -rf /var/lib/apt/lists/*

ENV RUSTUP_HOME=/usr/local/rustup
ENV CARGO_HOME=/usr/local/cargo
ENV PATH=/usr/local/cargo/bin:$PATH
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path --profile minimal
RUN cargo install --locked cbindgen

ENV LIBTORCH=/phantora/pytorch/torch
ENV LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
COPY Makefile Makefile
COPY phantora phantora
COPY include include
COPY stub stub
COPY dist dist
RUN mkdir build && make dist

RUN ln -sf bash /bin/sh
RUN ln -sf bash /usr/bin/sh
