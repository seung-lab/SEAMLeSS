# SEAMLeSS
SEAMLeSS (under active development)

Working repository for active development of the SEAMLeSS architecture for the purposes of aligning Basil. This codebase is 
not intended to remain consistent with the 2018 NIPS submission "Siamese Encoding and Alignment by Multiscale Learning with 
Self-Supervision" as it develops.

## Generating Data

Run `gen_stack.py` as `python gen_stack.py --count NUMBER_OF_SAMPLES --source CLOUD_VOLUME_PATH DATASET_NAME`, for example 
`python gen_stack.py --count 100 --source basil_v0/raw_image basil_v0`. This creates a training dataset of shape (100, 50, 
1152, 1152) at mip level 5, that is, 100 stacks of 50 slices of size 1152 x 1152. To change the number of slices per stack, 
image size, or mip level, pass the `--stack_height`, `--dim`, or `--mip` arguments. To generate the corresponding test set, 
pass the `--test` flag; samples are partitioned such that train samples come from (1 <= z < 800) and test samples come from 
(800 <= z < 1000).

## Training

`train.py` trains a pyramid. There are many arguments available to customize the training process, which are more or less 
self-documented within `train.py`. (TODO: document them here)

## Testing

To test a trained pyramid, run `python train.py --inference_only --state_archive pt/YOUR_ARCHIVE.pt NAME_OF_RUN`.
