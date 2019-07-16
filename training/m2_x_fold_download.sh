#!/bin/bash
STACK_SIZE=500
for Z_START in `seq 20500 $STACK_SIZE 20504`;
do
    for XY_START in `seq 180000 10000 180001`;
    do
        time python3 cv_sampler.py /usr/people/popovych/seungmount/research/sergiy/minnie_x_coarse_mip2/fold_mask_minnie_x_coarse_mip2_4096px_train_xy${XY_START}_z${Z_START} --source gs://seunglab2/minnie_v1/alignment/coarse/sergiy_multimodel_foldmask/image --mip 4 --stack_height $STACK_SIZE --xy_dim 1024 --start_point ${XY_START},${XY_START},${Z_START}
    done
done
