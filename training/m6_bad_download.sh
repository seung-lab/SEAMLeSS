#!/bin/bash
STACK_SIZE=1000
for Z_START in `seq 19000 $STACK_SIZE 19501`;
do
    time python3 cv_sampler.py /usr/people/popovych/seungmount/research/sergiy/minnie_v4_bad_mip6/minnie_v4_bad_mip6_2048px_train_z${Z_START} --source microns-seunglab/aibs-ingest/minnie/aibs_image_global_final_redo --mip 6 --stack_height $STACK_SIZE --xy_dim 2048 --start_point 180000,180000,${Z_START}
done
