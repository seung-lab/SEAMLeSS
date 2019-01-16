# Aligning CloudVolumes with SEAMLeSS  
This directory contains the scripts necessary to align a dataset stored as a
CloudVolume.

## Overview  
The SEAMLeSS models are trained to generate a vector field that warps a source
section to a target section. To produce an aligned dataset requires generating 
vector fields for pairs of sections as various MIP levels, manipulating those 
vector fields so they produce an aligned dataset, then using those final vector 
fields to render the dataset into the aligned space.  

Due to memory constraints, these models are developed to operate on images that 
are much smaller than typical CloudVolume datasets. To operate on a CloudVolume 
dataset requires chunk-based processing. The Aligner class in [aligner.py](aligner.py) 
coordinates this chunk-based processing for each script. It will download image 
chunks from all input CloudVolumes, apply the script, then upload any outputs 
back to their own CloudVolumes.

## Parameters  

## Workflows  

## Distributed operation  
SEAMLeSS has been designed for distributed operation across a cluster. We have 
currently optimized it for operation using: 

* [Google Cloud's Kubernetes Engine](https://console.cloud.google.com/kubernetes)
* [AWS SQS](https://console.aws.amazon.com/sqs/)

Here are some general guidelines on setting up a cluster to process SEAMLeSS tasks.

### Credentials  
You will need credentials for both Google Cloud & AWS. See the 
[CloudVolume README](https://github.com/seung-lab/cloud-volume#credentials) 
for instructions on obtaining them, and storing them in JSON format.

### Setting up Kubernetes  
* Install [kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl/) 
on your local workstation.
* Create a GPU cluster. We recommend the following parameters (based on the `GPU Accelerated Computing` template in GKE):

```
Master version >= 1.10.9-gke.5 (default) 
Two node pools:
1. micro pool
   Number of nodes: 3
   Machine type: f1-micro
2. gpu pool
   Number of nodes: as many as you like
   Container-Optimized OS (cos) (default)
   Cores: 1 vCPU
   Memory: 3.75 GB
   Number of GPUs: 1
   GPU type: NVIDIA Tesla K80
   Boot disk size: 16 GB
   Access scopes: 'Allow full access to all Cloud APIs'
   Enable preemptible nodes: True

Advanced options
* VPC-native, Enable VPC-native: True
```

* Connect to your cluster.  

Once your cluster has been created, within the console, select the `Connect` button to access the `kubectl` commands to connect your local workstation to your cluster. Now you can mount your secrets to your cluster nodes:

```
gcloud container clusters get-credentials your-cluster-name 
```

* Create secrets for your cluster.  

Your service account credentials for gcloud as a JSON are needed:  

```
kubectl create secret generic secrets --from-file=$HOME/.cloudvolume/secrets/google-secret.json
```

As are your AWS credentials, which are passed through as a [YAML file](https://kubernetes.io/docs/concepts/configuration/secret/#creating-a-secret-manually).

* Create a deployment for your cluster.  

See an example deployment in [docker/deploy.yaml](docker/deploy.yaml). You 
can deploy it with

```
kubectl create -f deploy.yaml
```

#### Further information  
* You can track your deployment in the browser with Google Console [Workloads](https://console.cloud.google.com/kubernetes/workload).
* You can also track your deployment from the command line with `kubectl`. See the [kubectl cheat sheet](https://kubernetes.io/docs/user-guide/kubectl-cheatsheet/) for commands.  

