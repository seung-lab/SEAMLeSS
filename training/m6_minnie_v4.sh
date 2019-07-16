#!/bin/bash
STACK_SIZE=200
#next: start from 14800
for Z_START in `seq 8000 $STACK_SIZE 10000`;
do
    FILE=/usr/people/popovych/seungmount/research/sergiy/minnie_v4_mip6/minnie_v4_full_mip6_6144px_train_z${Z_START}.h5
    #if ! test -f "$FILE"; then
        echo "$FILE doesnt exist. Downloading..."
        time python3 cv_sampler.py /usr/people/popovych/seungmount/research/sergiy/minnie_v4_mip6/minnie_v4_full_mip6_6144px_train_z${Z_START} --source microns-seunglab/minnie_v4/raw --mip 6 --stack_height $STACK_SIZE --xy_dim 6144 --start_point 30000,40000,${Z_START}
        echo "Done!"
    #fi
done
