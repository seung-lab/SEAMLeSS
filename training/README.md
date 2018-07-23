# SEAMLeSS

Working repository for active development of the SEAMLeSS architecture for the purposes of aligning Basil. This codebase is 
not intended to remain consistent with the 2018 NIPS submission "Siamese Encoding and Alignment by Multiscale Learning with 
Self-Supervision."

## Training

### Key Arguments
To control training, it is best to use arguments/parameters so that they are saved and archived by argparse. To see a list of all of the parameters available in training, run

`python train.py -h`

This will also show you a description for the purpose of each parameter.

The key parameters are the `lambda` family:

`lambda1` controls the weighting of the total smoothness penalty against the similarity penalty (this is the most important/frequently tuned parameter)

`lambda2` controls the relative weight of the smoothness penalty in the region near, but not within defects (cracks and folds); this is usually a value in the range (0.01,0.1)

`lambda3` controls the relative weight of the smoothness penalty in the regions *within* defects; this is almost always 0

`lambda4` is like lambda2, but for the similarity penalty; it is usually a value in the range (5,10), which focuses the net on fixing the defects

`lambda5` is like lambda3, but for the similarity penalty; it is almost always 0

`lambda6` is the coefficient of the consensus penalty used for eliminating drift; it is usually in the range 0.1-10 (this requires more investigation)

Some other key parameters:

`mask_neighborhood_radius` controls the radius in pixels of the neighborhood regions used by `lambda2` and `lambda4`

`lr` controls the learning rate; usually ~0.0003

`--state_archive` is a filepath to the network archive you'd like to start training from

### Invoking Training

You can begin training or fine-tuning by calling

`python train.py [--param1_name VALUE1 --param2_name VALUE2 ...] EXPERIMENT_NAME`

**If you use the same experiment name twice, you will overwrite the outputs from the older version. It is highly recommended that you use unique experiment names.**

### Fine-tuning
An example invocation of training to fine-tune a network called 'matriarch_na3' would be:

`python train.py --state_archive pt/matriarch_na3.pt --size 8 --lambda1 2 --lambda2 0.04 --lambda3 0 --lambda4 5 --lambda5 0 --mask_smooth_radius 75 --mask_neighborhood_radius 75 --lr 0.0003 --trunc 0 --fine_tuning --hm --padding 0 --vis_interval 5 --lambda6 1 fine_tune_example`

### Training from scratch (TODO)


## Generating Data

Run `gen_stack.py` as `python gen_stack.py --count NUMBER_OF_SAMPLES --source CLOUD_VOLUME_PATH DATASET_NAME`, for example 
`python gen_stack.py --count 100 --source basil_v0/raw_image basil_v0`. This creates a training dataset of shape (100, 50, 
1152, 1152) at mip level 5, that is, 100 stacks of 50 slices of size 1152 x 1152. To change the number of slices per stack, 
image size, or mip level, pass the `--stack_height`, `--dim`, or `--mip` arguments. To generate the corresponding test set, 
pass the `--test` flag; samples are partitioned such that train samples come from (1 <= z < 800) and test samples come from 
(800 <= z < 1000).

## Network Histories
Using the net_hist tool (`net_hist.py`), you can see the history of training for a particular network.

