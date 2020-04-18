#!/usr/bin/env python3

# python whatever --stack_height 2 --image_dim 1536 --coords coords_train.txt --image_src precomputed://gs://seunglab_minnie_phase3/alignment/precoarse_vv5_tempdiv4_step128_maxdisp128/warped_result --image_mip 2 --field_src precomputed://gs://seunglab_minnie_phase3/alignment/coarse_v1/inference_x2_optim300_lr3em2_sm25e1_maskthresh07_mse3_vv5_long/field/stitch/compose --field_mip 8 --mask_src precomputed://gs://seunglab_minnie_phase3/alignment/precoarse_vv5_tempdiv4_step128_maxdisp128/warped_folds/image --mask_mip 4 --coords coords_train.txt

# import numpy as np
import h5py
import sys
sys.path.append('./inference')

import argparse
import ast
import torch
from aligner import Aligner
from boundingbox import BoundingBox
from copy import deepcopy
from cloudmanager import CloudManager
from utilities.helpers import upsample_field
import torch.nn as nn

from PIL import Image
import numpy as np

name = sys.argv[1]
parser = argparse.ArgumentParser()
parser.add_argument('name')
# parser.add_argument('--count', type=int)
parser.add_argument('--stack_height', type=int, default=2)
parser.add_argument('--image_dim', type=int, default=1536)
parser.add_argument('--coords', type=str, default='coords_train.txt')
parser.add_argument('--image_src', type=str, default="precomputed://gs://seunglab_minnie_phase3/alignment/precoarse_vv5_tempdiv4_step128_maxdisp128/warped_result_mip2")
parser.add_argument('--image_mip', type=int, default=2)
parser.add_argument('--field_src', type=str, default="precomputed://gs://seunglab_minnie_phase3/alignment/coarse_v1/inference_x2_optim300_lr3em2_sm25e1_maskthresh07_mse3_vv5_long/field/stitch/compose")
parser.add_argument('--field_mip', type=int, default=8)
parser.add_argument('--mask_src', type=str, default="precomputed://gs://seunglab_minnie_phase3/alignment/precoarse_vv5_tempdiv4_step128_maxdisp128/warped_folds/image")
parser.add_argument('--mask_mip', type=int, default=4)
args = parser.parse_args()
print(args)


with open(args.coords) as coord_file:
    coords = ast.literal_eval(coord_file.read())

aligner = Aligner()

cm_image = CloudManager(args.image_src, 12, 0, {}, 1, size_chunk=args.image_dim, batch_mip=args.image_mip, create_info=False)
cm_mask = CloudManager(args.mask_src, 12, 0, {}, 1, size_chunk=args.image_dim, batch_mip=args.mask_mip, create_info=False)
cm_field = CloudManager(args.field_src, 12, 0, {}, 1, size_chunk=args.image_dim, batch_mip=args.field_mip, create_info=False)

cv_image = cm_image.create(args.image_src, data_type='uint8', num_channels=1, fill_missing=True, overwrite=False)
cv_mask = cm_mask.create(args.mask_src, data_type='uint8', num_channels=1, fill_missing=True, overwrite=False)
cv_field = cm_field.create(args.field_src, data_type='uint16', num_channels=2, fill_missing=True, overwrite=False)

coords = coords[:10]

sample_count = len(coords)
image_dim = args.image_dim
mask_dim = image_dim // (2 ** (args.mask_mip - args.image_mip))

h5f = h5py.File(args.name + '_train.h5', 'w')
h5f.create_dataset('images',
    (sample_count, args.stack_height, image_dim, image_dim),
    chunks=(1, 1, image_dim, image_dim),
    dtype=np.uint8,
    compression="lzf"
)
h5f.create_dataset('masks',
    (sample_count, args.stack_height, mask_dim, mask_dim),
    chunks=(1, 1, mask_dim, mask_dim),
    dtype=np.uint8,
    compression="lzf"
)
h5f.create_dataset('fields',
    (sample_count, args.stack_height, image_dim, image_dim, 2),
    chunks=(1, 1, image_dim, image_dim, 2),
    dtype=np.float32,
    compression="lzf",
    scaleoffset=2
)

for sample_id, coord in enumerate(coords):
    print(sample_id, "/", len(coords))
    xs, ys, zs = coord[:3]
    xs = round(xs / 2**args.mask_mip) * 2**args.mask_mip
    ys = round(ys / 2**args.mask_mip) * 2**args.mask_mip
    xs_snap = round(xs / 2**args.field_mip) * 2**args.field_mip
    ys_snap = round(ys / 2**args.field_mip) * 2**args.field_mip

    xy_size, z_size = coord[-2], coord[-1]
    bbox = BoundingBox(xs_snap, xs_snap + xy_size, ys_snap, ys_snap + xy_size, mip=0, max_mip=8)

    padded_bbox = deepcopy(bbox)
    padded_bbox.uncrop(2**args.field_mip, 0)

    crop = 2 ** (args.field_mip - args.image_mip)
    xs_offset = (xs_snap - xs) // 2**args.image_mip
    ys_offset = (ys_snap - ys) // 2**args.image_mip

    for z_off in range(args.stack_height):
        # Get coarse field
        coarse_field = aligner.get_field(cv_field, zs + z_off, padded_bbox, args.field_mip, relative=False, to_tensor=True, as_int16=True)
        coarse_field_up = coarse_field
        coarse_field_up = coarse_field_up.permute(0, 3, 1, 2)
        coarse_field_up = nn.Upsample(scale_factor=2**(args.field_mip - args.image_mip), mode='bilinear')(coarse_field_up)
        coarse_field_up = coarse_field_up.permute(0, 2, 3, 1)
        #coarse_field_up = upsample_field(coarse_field, src_mip=args.field_mip, dst_mip=args.image_mip)
        coarse_field_up = coarse_field_up[:, crop+xs_offset:-crop+xs_offset, crop+ys_offset:-crop+ys_offset, :]

        # Average displacement and calculate relevant image patch
        mean_disp = aligner.profile_field(coarse_field_up)
        mean_disp = torch.round(mean_disp / 2**args.mask_mip) * 2**args.mask_mip
        displaced_bbox = aligner.adjust_bbox(bbox, mean_disp.flip(0))
        h5f['fields'][sample_id, z_off, :, :, :] = (coarse_field_up - mean_disp.to(device='cuda'))[0].cpu().numpy()

        # Get image + mask patch

        h5f['images'][sample_id, z_off, :, :] = np.uint8(255.0 * aligner.get_image(cv_image, zs + z_off, displaced_bbox, args.image_mip, to_tensor=True).cpu().numpy())
        h5f['masks'][sample_id, z_off, :, :] = np.uint8(255.0 * aligner.get_image(cv_mask, zs + z_off, displaced_bbox, args.mask_mip, to_tensor=True).cpu().numpy())

        # fx_min = np.min(h5f['fields'][sample_id, z_off, :, :, 0])
        # fx_max = np.max(h5f['fields'][sample_id, z_off, :, :, 0])
        # fy_min = np.min(h5f['fields'][sample_id, z_off, :, :, 1])
        # fy_max = np.max(h5f['fields'][sample_id, z_off, :, :, 1])
        # Image.fromarray(np.uint8(255.0 * (h5f['fields'][sample_id, z_off, :, :, 0] - fx_min) / (fx_max - fx_min))).save(f"test/{sample_id}_{z_off}_field_x.png")
        # Image.fromarray(np.uint8(255.0 * (h5f['fields'][sample_id, z_off, :, :, 1] - fy_min) / (fy_max - fy_min))).save(f"test/{sample_id}_{z_off}_field_y.png")

        # Image.fromarray(h5f['images'][sample_id, z_off, :, :]).save(f"test/{sample_id}_{z_off}_image.png")
        # Image.fromarray(h5f['masks'][sample_id, z_off, :, :]).save(f"test/{sample_id}_{z_off}_mask.png")

h5f.close()

