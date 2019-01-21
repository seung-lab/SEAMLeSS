#!/bin/sh
# connect to cluster
gcloud container clusters get-credentials $1 --zone us-central1-a --project iarpa-microns-seunglab
#install NVIDIA GPU device drivers
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/stable/nvidia-driver-installer/cos/daemonset-preloaded.yaml
#create secrets for AWS and Google cloud
kubectl create secret generic secrets --from-file=/usr/people/tmacrina/.cloudvolume/secrets/google-secret.json --from-file=/usr/people/tmacrina/.cloudvolume/secrets/seunglab2-google-secret.json --from-file=/usr/people/tmacrina/.cloudvolume/secrets/microns-seunglab-google-secret.json
kubectl create -f secret.yaml 
kubectl create -f deploy.yaml
# scale deployment 
# kubectl scale deployment seamless --replicas=64
