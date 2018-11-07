#!/bin/bash
for mip in `seq 10 10`; do
    time python gen_stack.py minnie_nobord_train --source neuroglancer/minnie_v0/raw_image_2 --mip ${mip} --stack_height 400 --count 1 --xs 121176 --ys 149696 --zs 1900 --xe 220146 --ye 220106 --ze 2301 --dim 160 --no_split
    time python gen_stack.py minnie_nobord_val --source neuroglancer/minnie_v0/raw_image_2 --mip ${mip} --stack_height 50 --count 1 --xs 121176 --ys 149696 --zs 2300 --xe 220146 --ye 220106 --ze 2350 --dim 160 --no_split
done
#python gen_stack.py --count 1 --mip ${mip}  --stack_height 100 --dim 68 --source neuroglancer/seamless/vector_fixer30_fine_tuning_low_mip_e32_t0__pinky_downsampled_400sections/image --zs 1300 --ze 1401 --xs 55014 --xe 80830 --ys 37747 --ye 65563 pinky_all_mips_unsup_val --no_split
