#!/bin/bash
gcloud beta container --project "iarpa-microns-seunglab" clusters create "gpu-cluster-1" --zone "us-central1-a" --username "admin" --cluster-version "1.11.6-gke.2" --machine-type "n1-standard-2" --image-type "COS" --disk-type "pd-standard" --disk-size "10" --scopes
"https://www.googleapis.com/auth/devstorage.read_only","https://www.googleapis.com/auth/logging.write","https://www.googleapis.com/auth/monitoring","https://www.googleapis.com/auth/servicecontrol","https://www.googleapis.com/auth/service.management.readonly","https://www.googleapis.com/auth/trace.append" --num-nodes "3" --enable-cloud-logging --enable-cloud-monitoring --enable-ip-alias --network "projects/iarpa-microns-seunglab/global/networks/seamless-sergiy"
--subnetwork "projects/iarpa-microns-seunglab/regions/us-central1/subnetworks/seamless-sergiy-subnet" --default-max-pods-per-node "8" --addons HorizontalPodAutoscaling,HttpLoadBalancing --enable-autoupgrade --enable-autorepair && gcloud beta container --project "iarpa-microns-seunglab" node-pools create "gpu-pool-1" --cluster "gpu-cluster-1" --zone "us-central1-a" --node-version "1.11.6-gke.2" --machine-type "n1-highmem-2" --accelerator "type=nvidia-tesla-t4,count=1"
--image-type "COS" --disk-type "pd-standard" --disk-size "10" --scopes "https://www.googleapis.com/auth/cloud-platform" --preemptible --num-nodes "1" --enable-autoupgrade --enable-autorepair

