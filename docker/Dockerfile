# Reference:
# https://github.com/cvpaperchallenge/Ascender
# https://github.com/nerfstudio-project/nerfstudio

FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel

ARG USER_NAME=cv

# apt install by root user
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libegl1-mesa-dev \
    libgl1-mesa-dev \
    libgles2-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    python-is-python3 \
    python3.10-dev \
    python3-pip \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip setuptools ninja
RUN pip install -r /tmp/requirements.txt

# Change user to non-root user
RUN adduser --disabled-password --gecos "" ${USER_NAME}
RUN chown -R ${USER_NAME}:${USER_NAME} /home/${USER_NAME}
COPY . /home/${USER_NAME}/image-restoration
RUN mkdir /home/${USER_NAME}/checkpoints
RUN chown -R ${USER_NAME}:${USER_NAME} /home/${USER_NAME}/checkpoints
USER ${USER_NAME}
WORKDIR /home/${USER_NAME}/image-restoration
