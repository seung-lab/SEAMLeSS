#!/bin/bash
Z_START=8000
time python3 cv_sampler.py /usr/people/popovych/seungmount/research/sergiy/minnie_dynamic/minnie_dynamic_full_mip6_6144px_train_z${Z_START} --source seunglab_alembic/minnie/aibs_image_globalalign --mip 6 --stack_height 200 --xy_dim 6144 --start_point 80000,90000,${Z_START}
