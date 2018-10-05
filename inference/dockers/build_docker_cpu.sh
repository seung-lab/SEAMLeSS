#!/bin/bash
docker build -t seunglab/nflow_inference:cpu -f Dockerfile.cpu .
docker push seunglab/nflow_inference:cpu
