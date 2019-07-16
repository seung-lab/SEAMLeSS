#!/bin/bash
STACK_SIZE=500
for Z_START in `seq 20500 $STACK_SIZE 20504`;
do
    for XY_START in `seq 200000 10000 200001`;
    do
        time python3 cv_sampler.py /usr/people/popovych/seungmount/research/sergiy/minnie_x_coarse_mip4/minnie_x_coarse_mip4_2048px_train_xy${XY_START}_z${Z_START} --source gs://microns-seunglab/minnie_v3/alignment/coarse/sergiy_multimodel_v5/boss_mip2 --mip 4 --stack_height $STACK_SIZE --xy_dim 2048 --start_point ${XY_START},${XY_START},${Z_START}
    done
done
