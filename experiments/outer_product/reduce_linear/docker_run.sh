#!/bin/bash

docker run -it --rm -v $("pwd"):/home --gpus all --ipc="host" jax_workbench:latest "$@"
