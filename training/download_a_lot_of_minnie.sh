#!/bin/bash
for Z_START in `seq 20800 200 22000`;
do
    time python3 cv_sampler.py /usr/people/popovych/seungmount/research/sergiy/minnie_mip6/minnie_v1_full_mip6_6144px_train_z${Z_START} --source seunglab2/minnie_v1/raw_image --mip 6 --stack_height 200 --xy_dim 6144 --start_point 80000,90000,${Z_START}
done
