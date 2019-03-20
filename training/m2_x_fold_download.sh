#!/bin/bash
STACK_SIZE=500
for Z_START in `seq 20000 $STACK_SIZE 20002`;
do
    time python3 cv_sampler.py /usr/people/popovych/seungmount/research/sergiy/minnie_x_mip2_coarse/fold_mask_minnie_v4_coarse_mip2_2048px_train_z${Z_START} --source gs://seunglab2/minnie_v1/alignment/coarse/sergiy_multimodel_foldmask/image --mip 4 --stack_height $STACK_SIZE --xy_dim 512 --start_point 180000,180000,${Z_START}
done
