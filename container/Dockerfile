ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.05-py3

FROM ${BASE_IMAGE}

# APT packages
RUN apt-get update && \
  apt-get install -y bc gdb libsox-fmt-all nvtop -y && \
  apt clean

RUN pip install \
    wandb==0.19.5 \
    huggingface-hub==0.28.1 \
    transformers==4.48.2 \
    tokenizers==0.21.0 \
    sentencepiece==0.2.0 \
    tiktoken==0.8.0 \
    aniso8601==10.0.0 \
    blinker==1.9.0 \
    itsdangerous==2.2.0
