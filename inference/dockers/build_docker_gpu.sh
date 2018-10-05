#!/bin/bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
docker build -t seunglab/nflow_inference:gpu -f Dockerfile.gpu ../../
docker push seunglab/nflow_inference:gpu
