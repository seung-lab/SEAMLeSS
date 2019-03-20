#!/bin/bash
STACK_SIZE=500
for Z_START in `seq 20000 $STACK_SIZE 20002`;
do
    time python3 cv_sampler.py /usr/people/popovych/seungmount/research/sergiy/minnie_x_mip2_coarse/minnie_v4_coarse_mip2_2048px_train_z${Z_START} --source gs://microns-seunglab/minnie_v3/alignment/coarse/sergiy_multimodel_v5/upsampled_image --mip 2 --stack_height $STACK_SIZE --xy_dim 2048 --start_point 180000,180000,${Z_START}
done
