#FROM nvidia/cuda:11.4.1-cudnn8-runtime-ubuntu20.04
FROM nvidia/cuda:11.6.0-devel-ubuntu20.04

# declare the image name
ENV IMG_NAME=11.6.0-devel-ubuntu20.04 \
    # declare what jaxlib tag to use
    # if a CI/CD system is expected to pass in these arguments
    # the dockerfile should be modified accordingly
    JAXLIB_VERSION=0.3.2

RUN apt update && apt install python3-pip git -y

RUN pip3 install git+https://github.com/ptmorris03/jaks.git
RUN pip3 install jaxlib==${JAXLIB_VERSION}+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_releases.html jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_releases.html
RUN pip3 install imageio scikit-learn einops optax

WORKDIR /home

ENTRYPOINT ["python3"]
