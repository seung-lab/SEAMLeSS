#!/bin/bash
STACK_SIZE=200
for Z_START in `seq 20000 $STACK_SIZE 20002`;
do
    time python3 cv_sampler.py /usr/people/popovych/seungmount/research/sergiy/minnie_v4_mip2_coarse/minnie_v4_coarse_mip2_4096px_train_z${Z_START} --source gs://microns-seunglab/minnie_v4/alignment/coarse/sergiy_multimodel_v1/mip2 --mip 2 --stack_height $STACK_SIZE --xy_dim 4096 --start_point 170000,170000,${Z_START}
done
