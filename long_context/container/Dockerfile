ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.01-py3
FROM ${BASE_IMAGE}
# APT packages
RUN apt-get update && \
  apt-get install -y bc gdb libsox-fmt-all -y && \
  apt-get clean
RUN pip install flask-restful einops sentencepiece transformers tiktoken wandb