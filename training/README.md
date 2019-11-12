# SEAMLeSS Training
Training code for SEAMLeSS networks.

## Getting started
Training requires a training set. 
See the [README in data_handling](data_handling/README.md) for more details.

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
