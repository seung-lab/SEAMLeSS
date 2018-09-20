#!/bin/bash

python worker.py --model_path model_repository/vector_fixer30_fine_tuning_low_mip_e9_t200_.pt --mip 2 --max_mip 6 --render_mip 2 --num_targets 2 --xs 0 --xe 69000 --ys 0 --ye 44000 --zs 357 --stack_size 20 --source gs://neuroglancer/pinky40_alignment/prealigned_rechunked --edge_pad 256 --should_contrast 1 --max_displacement 2048 --size 8 --out_name prealigned_sergiy_error_handler --queue_name deepalign
