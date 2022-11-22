FROM python:3.10-slim-buster

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Oslo
ENV DOCKERUSER=docker_user

RUN apt-get update && apt-get install -y \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
RUN useradd -m -s /bin/bash ${DOCKERUSER}
USER ${DOCKERUSER}}
ENV PATH "$PATH:/home/${DOCKERUSER}/.local/bin"

USER root

COPY ./requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/home/${DOCKERUSER}/hugging_face:"

USER ${DOCKERUSER}
WORKDIR /home/${DOCKERUSER}/hugging_face
ENTRYPOINT ["bash"]