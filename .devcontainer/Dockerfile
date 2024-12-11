# If .venv is already setup with python3.8, it will use python3.8. To use 3.11 remove it first.

# Use Nvidia Ubuntu 20 base (includes CUDA if a supported GPU is present)
# https://hub.docker.com/r/nvidia/cuda
FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04@sha256:55211df43bf393d3393559d5ab53283d4ebc3943d802b04546a24f3345825bd9

ARG USERNAME
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
# https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user
RUN groupadd --gid $USER_GID $USERNAME \
  && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
  && usermod -a -G video user \ 
  && apt-get update \
  && apt-get install -y sudo \
  && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
  && chmod 0440 /etc/sudoers.d/$USERNAME

# Install dependencies
RUN apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get -qq -y install \
  software-properties-common && \
  add-apt-repository -y ppa:deadsnakes/ppa && \
  apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get -qq -y install \
  build-essential \
  python3.11 \
  python3.11-dev \
  python3.11-distutils \
  python3.11-venv \
  curl \
  git && \
  # Update python3 default to point to python3.11
  update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
  update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2 && \
  update-alternatives --set python3 /usr/bin/python3.11

# User the new user
USER $USERNAME

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python3.11 -
