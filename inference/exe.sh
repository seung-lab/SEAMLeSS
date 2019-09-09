date
CUDA_VISIBLE_DEVICES=1 python refactor_align_blocks.py --model_lookup params/test_whole.csv  --src_path gs://microns-seunglab/minnie_v3/alignment/coarse/sergiy_multimodel_v5/upsampled_image  --dst_path gs://microns-seunglab/zhen_test/refactor_mip2_test  --mip 2 --max_mip 6  --tgt_radius 5 --pad 256  --block_size 10  --z_start 20795 --z_stop 20805

