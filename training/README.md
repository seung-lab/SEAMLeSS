# SEAMLeSS

Working repository for active development of the SEAMLeSS architecture for the purposes of aligning Basil. This codebase is 
not intended to remain consistent with the 2018 NIPS submission "Siamese Encoding and Alignment by Multiscale Learning with 
Self-Supervision."

## Generating Data

Run `gen_stack.py` as `python gen_stack.py --count NUMBER_OF_SAMPLES --source CLOUD_VOLUME_PATH DATASET_NAME`, for example 
`python gen_stack.py --count 100 --source basil_v0/raw_image basil_v0`. This creates a training dataset of shape (100, 50, 
1152, 1152) at mip level 5, that is, 100 stacks of 50 slices of size 1152 x 1152. To change the number of slices per stack, 
image size, or mip level, pass the `--stack_height`, `--dim`, or `--mip` arguments. To generate the corresponding test set, 
pass the `--test` flag; samples are partitioned such that train samples come from (1 <= z < 800) and test samples come from 
(800 <= z < 1000).

## Training

An example invocation of training to fine-tune a network called 'matriarch_na3' would be:

`python train.py --state_archive pt/matriarch_na3.pt --size 8 --lambda1 2 --lambda2 0.04 --lambda3 0 --lambda4 5 --lambda5 0 --mask_smooth_radius 75 --mask_neighborhood_radius 75 --lr 0.0003 --trunc 0 --fine_tuning --hm --padding 0 --vis_interval 5 --lambda6 1 fine_tune_example`

## Testing

To test a trained pyramid, run `python train.py --inference_only --state_archive pt/YOUR_ARCHIVE.pt NAME_OF_RUN`.

