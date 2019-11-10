# SEAMLeSS Training
Training code for SEAMLeSS networks.

## Getting started
Training requires a training set. 
A training set is an H5 file, containing at least one dataset. 
Each dataset is a stack of images stored as a 4D array (1xZxWxH). 
Images in the stack should be sequential along the Z dimension.

To train a network from scratch with default parameters, use the following
command:
```
python train.py start NAME --training_set PATH_TO_DATASET
```
To learn more about the training parameters, use the `--help` flag.
```
python train.py start --help 
```

To resume training a network, use the following command:
```
python train.py resume NAME
```

## Parallelilzation
Training is parallelized by default (both across GPUs and CPUs for gradient
descent and data loading). To learn more about the parallelization parameters,
use the `--help` flag.
```
python train.py --help 
```

## Generating a training set
See the [README in data_handling](data_handling/README.md) for more details.
