#FROM nvidia/cuda:11.4.1-cudnn8-runtime-ubuntu20.04
FROM nvidia/cuda:11.6.0-runtime-ubuntu20.04

ENV JAXLIB_VERSION=0.3.0

RUN apt update && apt install python3-pip git -y

RUN pip3 install git+https://github.com/ptmorris03/jax_workbench.git
RUN pip3 install jaxlib==${JAXLIB_VERSION}+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_releases.html jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_releases.html

WORKDIR /home

ENTRYPOINT ["python3"]
