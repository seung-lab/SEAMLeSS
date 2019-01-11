#!/bin/bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
docker build -t seunglab/seamless:pairwise -f Dockerfile.gpu ../../
docker push seunglab/seamless:pairwise
