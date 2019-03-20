#!/bin/bash
STACK_SIZE=200
for Z_START in `seq 19000 $STACK_SIZE 19000`;
do
    time python3 cv_sampler.py /usr/people/popovych/seungmount/research/sergiy/minnie_v4_mip6/minnie_v4_full_mip6_6144px_train_z${Z_START} --source microns-seunglab/minnie_v4/raw --mip 6 --stack_height $STACK_SIZE --xy_dim 6144 --start_point 40000,90000,${Z_START}
done
