#!/bin/bash
CHUNK=1024
for Z_START in `seq 8000 $CHUNK 27000`;
do
    time python3 cv_sampler.py /usr/people/popovych/seungmount/research/sergiy/minnie_v2_mip8/minnie_v2_full_mip8_6144px_train_z${Z_START} --source microns-seunglab/minnie_v2/raw --mip 8 --stack_height $CHUNK --xy_dim 1536 --start_point 80000,90000,${Z_START} &
done
