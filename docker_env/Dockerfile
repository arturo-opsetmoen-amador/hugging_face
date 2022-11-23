FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Oslo
ENV DOCKERUSER=docker_user
ARG PYTHON_VERSION=3.7.15


RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    yasm \
    pkg-config \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavformat-dev \
    libpq-dev \
    libxine2-dev \
    libglew-dev \
    libtiff5-dev \
    zlib1g-dev \
    libjpeg-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libpostproc-dev \
    libswscale-dev \
    libeigen3-dev \
    libtbb-dev \
    libgtk2.0-dev \
    libbz2-dev \
    pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz
RUN tar -xf Python-${PYTHON_VERSION}.tgz
RUN cd Python-${PYTHON_VERSION}/ && ./configure --enable-optimizations && make altinstall -j"$(nproc)"
RUN update-alternatives --install /usr/bin/python python /usr/local/bin/python3.7 1
RUN update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3.7 1

RUN pip install --upgrade pip
RUN useradd -m -s /bin/bash ${DOCKERUSER}
USER ${DOCKERUSER}
ENV PATH "$PATH:/home/${DOCKERUSER}/.local/bin"

USER root

COPY ./requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/home/${DOCKERUSER}/hugging_face:"

USER ${DOCKERUSER}
WORKDIR /home/${DOCKERUSER}/hugging_face
ENTRYPOINT ["bash"]
