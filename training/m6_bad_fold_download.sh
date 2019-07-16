#!/bin/bash
STACK_SIZE=1000

for Z_START in `seq 19000 $STACK_SIZE 19501`;
do
    time python3 cv_sampler.py /usr/people/popovych/seungmount/research/sergiy/minnie_v4_bad_mip6/fold_mask_minnie_v4_bad_mip6_2048px_train_z${Z_START} --source neuroglancer/alex/fold_detection/minnie/minnie_raw_foldmask/image --mip 6 --stack_height $STACK_SIZE --xy_dim 2048 --start_point 180000,180000,${Z_START}
done
