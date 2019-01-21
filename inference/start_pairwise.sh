#!/bin/bash

SRC_PATH="gs://seunglab2/minnie_v1/alignment/coarse/tmacrina_minnie10_serial/upsampled_image"
DST_PATH="gs://seunglab2/minnie_v1/alignment/fine/sergiy_pairwise"
XY_START="160000 300000"
XY_END="180000 320000"
Z_START=19900
SERIAL_Z_START=$(expr $Z_START - 3)
SERIAL_Z_END=$(expr $Z_START + 1)
MODEL="vector_fixer30"

echo $(expr $Z_START + 1)
echo python3 serial_alignment.py --model_path $MODEL --src_path $SRC_PATH --src_mask_path gs://seunglab2/minnie_v1/raw_image/resin_mask --src_mask_mip 8 --src_mask_val 1 --tgt_mask_path gs://seunglab2/minnie_v1/raw_image/resin_mask --tgt_mask_mip 8 --tgt_mask_val 1 --dst_path ${DST_PATH}_starter --mip 2 --max_mip 8 --render_low_mip 2 --render_high_mip 8 --bbox_start $XY_START $SERIAL_Z_START --bbox_stop $XY_END $SERIAL_Z_END --bbox_mip 0 --should_contrast --max_displacement 2048 --size 8 --disable_flip_average --old_vectors --tgt_radius 3 --no_vvote_start

echo python3 multi_match.py --model_path $MODEL--src_path ${DST_PATH}_starter/image --src_mask_path gs://seunglab2/minnie_v1/raw_image/resin_mask --src_mask_mip 8 --src_mask_val 1 --tgt_mask_path gs://seunglab2/minnie_v1/raw_image/resin_mask --tgt_mask_mip 8 --tgt_mask_val 1 --dst_path ${DST_PATH} --mip 2 --max_mip 8 --render_low_mip 2 --render_high_mip 8 --bbox_start $XY_START $Z_START --bbox_stop $XY_END $SERIAL_Z_END --bbox_mip 0 --should_contrast --max_displacement 2048 --size 8 --disable_flip_average --old_vectors --tgt_radius 3 --forward_match

echo python3 match_section.py --model_path $MODEL --src_path $SRC_PATH --src_mask_path gs://seunglab2/minnie_v1/raw_image/resin_mask --src_mask_mip 8 --src_mask_val 1 --tgt_path ${DST_PATH}_starter/image --dst_path ${DST_PATH} --mip 2 --max_mip 8 --render_low_mip 2 --render_high_mip 8 --bbox_start $XY_START $Z_START --bbox_stop $XY_END $(expr $Z_START + 1) --bbox_mip 0 --should_contrast --max_displacement 2048 --size 8 --disable_flip_average --old_vectors --tgt_radius 3 --src_z $(expr $Z_START + 1) --tgt_z $(expr $Z_START - 2)

echo python3 match_section.py --model_path $MODEL --src_path $SRC_PATH --src_mask_path gs://seunglab2/minnie_v1/raw_image/resin_mask --src_mask_mip 8 --src_mask_val 1 --tgt_path ${DST_PATH}_starter/image --dst_path ${DST_PATH} --mip 2 --max_mip 8 --render_low_mip 2 --render_high_mip 8 --bbox_start $XY_START $Z_START --bbox_stop $XY_END $(expr $Z_START + 1) --bbox_mip 0 --should_contrast --max_displacement 2048 --size 8 --disable_flip_average --old_vectors --tgt_radius 3 --src_z $(expr $Z_START + 1) --tgt_z $(expr $Z_START - 1)

echo python3 match_section.py --model_path $MODEL --src_path $SRC_PATH --src_mask_path gs://seunglab2/minnie_v1/raw_image/resin_mask --src_mask_mip 8 --src_mask_val 1 --tgt_path ${DST_PATH}_starter/image --dst_path ${DST_PATH} --mip 2 --max_mip 8 --render_low_mip 2 --render_high_mip 8 --bbox_start $XY_START $Z_START --bbox_stop $XY_END $(expr $Z_START + 1) --bbox_mip 0 --should_contrast --max_displacement 2048 --size 8 --disable_flip_average --old_vectors --tgt_radius 3 --src_z $(expr $Z_START + 2) --tgt_z $(expr $Z_START - 1)
