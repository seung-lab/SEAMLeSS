#! /bin/sh
#install NVIDIA GPU device drivers
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/stable/nvidia-driver-installer/cos/daemonset-preloaded.yaml
#create secrets for AWS and Google cloud
kubectl create -f secret.yaml
kubectl create secret generic secrets --from-file=/usr/people/zhenj/.cloudvolume/secrets/google-secret.json --from-file=/usr/people/zhenj/.cloudvolume/secrets/seunglab2-google-secret.json
#kubectl create -f deploy.yaml
