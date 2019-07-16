#!/bin/bash
STACK_SIZE=200
#next: start from 14800
for Z_START in `seq 0 $STACK_SIZE 999`;
do
    FILE=/usr/people/popovych/seungmount/research/sergiy/basil_foa_mip6/basil_foa_mip6_z${Z_START}.h5
    #if ! test -f "$FILE"; then
        echo "$FILE doesnt exist. Downloading..."
        time python3 cv_sampler.py ${FILE} --source neuroglancer/basil_v0/father_of_alignment/v3 --mip 6 --stack_height $STACK_SIZE --xy_dim 5120 --start_point 0,0,${Z_START}
        echo "Done!"
    #fi
done
