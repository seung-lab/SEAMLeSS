#!/bin/bash
CHUNK=1024
for Z_START in `seq 8000 $CHUNK 27000`;
do
    time python3 cv_sampler.py /home/popovych/datasets/minnie_mip8/minnie_v2_full_mip8_2048px_train_z${Z_START} --source microns-seunglab/minnie_v2/raw --mip 8 --stack_height $CHUNK --xy_dim 2048 --start_point 40000,40000,${Z_START}
done
