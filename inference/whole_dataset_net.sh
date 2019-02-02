#!/bin/bash
NET_NAME=$1
RUN=5
BLOCK_SIZE=100
time python3 serial_block_alignment.py --model_path ../models/${NET_NAME} --src_path gs://microns-seunglab/minnie_v2/raw --dst_path gs://microns-seunglab/minnie_v2/alignment/large_runs/block${BLOCK_SIZE}_${NET_NAME}_${RUN} --bbox_start 40000 10000 8400 --bbox_stop 460000 370000 9400 --bbox_mip 0 --max_mip 8 --max_displacement 4096 --mip 6 --block_size ${BLOCK_SIZE} --queue_name deepalign_sergiy


#python3 serial_block_broadcast.py --src_path gs://microns-seunglab/minnie_v2/raw --dst_path gs://microns-seunglab/minnie_v2/alignment/large_runs/block${BLOCK_SIZE}_${NET_NAME}_${RUN}/image --src_field gs://microns-seunglab/minnie_v2/alignment/large_runs/block${BLOCK_SIZE}_${NET_NAME}_${RUN}/field/vvote --dst_field gs://microns-seunglab/minnie_v2/alignment/large_runs/block${BLOCK_SIZE}_${NET_NAME}_${RUN}/field/block_composed2 --bbox_start 40000 10000 8000 --bbox_stop 460000 370000 26000 --bbox_mip 0 --max_mip 8 --max_displacement 4096 --mip 6 --block_size 100 --queue_name deepalign_sergiy

