#!/bin/bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
TAGNAME=$1
docker build -t seunglab/seamless:$TAGNAME -f Dockerfile.gpu ../../
docker push seunglab/seamless:$TAGNAME
