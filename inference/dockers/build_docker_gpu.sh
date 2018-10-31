#!/bin/bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
docker build -t seunglab/pariwise_inference:gpu_test -f Dockerfile.gpu ../../
docker push seunglab/pairwise_inference:gpu_test
