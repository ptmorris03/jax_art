FROM nvidia/cuda:11.4.1-cudnn8-runtime-ubuntu20.04

ENV JAXLIB_VERSION=0.3.0

RUN apt update && apt install python3-pip -y

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 jaxlib==${JAXLIB_VERSION}+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_releases.html jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_releases.html

WORKDIR /home

ENTRYPOINT ["python3"]
