#!/bin/bash
STACK_SIZE=200
#next: start from 14800
for Z_START in `seq 22500 $STACK_SIZE 23500`;
do
    FILE=/usr/people/popovych/seungmount/research/sergiy/minnie_v4_m6_normed/fold_mask_normed_fill_minnie_v4_m6_6144px_z${Z_START}
    #if ! test -f "$FILE"; then
        echo "$FILE doesnt exist. Downloading..."
        time python3 cv_sampler.py ${FILE} --source neuroglancer/alex/fold_detection/minnie/minnie_raw_foldmask/image --mip 6 --stack_height $STACK_SIZE --xy_dim 6144 --start_point 30720,40960,${Z_START}
        echo "Done!"
    #fi
done
