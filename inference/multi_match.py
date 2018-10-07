import sys
import torch
from eval.cpc import CPC
from aligner import Aligner, BoundingBox
from link_builder import ng_link
import argparse

def multi_match(aligner, bbox, z_range):
  aligner.total_bbox = bbox
  for tgt_z in z_range:
    src_z_list = list(set(z_range) - set([tgt_z]))
    aligner.multi_match(tgt_z, src_z_list)

def cpc(aligner, bbox, z_range, src_mip=4, dst_mip=8):
    src_path = aligner.orig_src_path
    bbox_start, bbox_stop = bbox.get_bounding_pts()
    bbox_start = [*bbox_start, z_range[0]]
    bbox_stop = [*bbox_stop, z_range[-1] + 1]
    max_offset = z_range[-1] - z_range[0] + 1
    for k in range(1, max_offset):
      for j in [-1, 1]:
        z_offset = k*j
        tgt_path = '{0}/z{1}/image'.format(aligner.orig_dst_path, z_offset)
        dst_path = '{0}/z{1}/cpc48'.format(aligner.orig_dst_path, z_offset)
        print('src_path {0}'.format(src_path))
        print('tgt_path {0}'.format(tgt_path))
        print('dst_path {0}'.format(dst_path))
        e = CPC(src_path, tgt_path, dst_path, src_mip, dst_mip, bbox_start,
                  bbox_stop, 0, 0, 0, False, False)
        e.run()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--src_path', type=str,
    help='CloudVolume path of images to be warped')
  parser.add_argument('--tgt_path', type=str,
    help='CloudVolume path of images to align against; default: dst_path')
  parser.add_argument('--dst_path', type=str,
    help='CloudVolume path of rendered images')
  parser.add_argument('--size', type=int, default=8)
  parser.add_argument('--skip', type=int, default=0)
  parser.add_argument('--model_path', type=str)
  parser.add_argument('--mip', type=int)
  parser.add_argument('--render_mip', type=int)
  parser.add_argument('--should_contrast', type=int)
  parser.add_argument('--num_targets', type=int, default=1)
  parser.add_argument('--edge_pad', type=int, default=384)
  parser.add_argument('--max_displacement', 
    help='the size of the largest displacement expected; should be 2^high_mip', 
    type=int, default=2048)
  parser.add_argument('--max_mip', type=int, default=9)
  parser.add_argument('--bbox_start', nargs=3, type=int,
    help='bbox origin, 3-element int list')
  parser.add_argument('--bbox_stop', nargs=3, type=int,
    help='bbox origin+shape, 3-element int list')
  parser.add_argument('--bbox_mip', type=int, default=0,
    help='MIP level at which bbox_start & bbox_stop are specified')
  parser.add_argument('--no_flip_average', 
    help='disable flip averaging, on by default (flip averaging is used to eliminate drift)', 
    action='store_true')
  parser.add_argument('--run_pairs', 
    help='only run on consecutive pairs of input slices, rather than sequentially aligning a whole stack', 
    action='store_true')
  parser.add_argument('--old_upsample', help='revert to the old pytorch upsampling (using align_corners=True)',
    action='store_true')
  parser.add_argument('--old_vectors', help='expect the net to return vectors in the old vector field convention, '
    'where -1 and 1 refer to the centers of the border pixels rather than the image edges.',
    action='store_true')
  parser.add_argument('--ignore_field_init', help='do not initialize the field cloudvolume (already exists)',
    action='store_true')
  parser.add_argument('--write_intermediaries', 
    help='write encodings, residuals, & cumulative residuals to cloudvolumes', 
    action='store_true')
  parser.add_argument('--upsample_residuals', 
    help='upsample residuals & cumulative residuals when writing intermediaries; requires --write_intermediaries flag', 
    action='store_true')
  parser.add_argument('--z_offset', type=int, default=1,
    help='Offset in z for target slice')
  args = parser.parse_args()
  
  args.tgt_path = args.tgt_path if args.tgt_path else args.src_path
 
  mip = args.mip
  render_mip = args.render_mip
  should_contrast = bool(args.should_contrast)
  num_targets = args.num_targets
  model_path = args.model_path
  edge_pad  = args.edge_pad
  mip_range = (mip,mip)
  high_mip_chunk = (1024, 1024)
  max_mip = args.max_mip
  
  print('model_path: {0}'.format(args.model_path))
  print('src_path: {0}'.format(args.src_path))
  print('dst_path: {0}'.format(args.dst_path))
  print('Coordinates:', args.bbox_start, args.bbox_stop)
  print('Mip:', mip)
  print('Contrast:', should_contrast)
  print('Max mip:', max_mip)
  print('NG link:', ng_link('dst', 'precomputed://' + args.dst_path +'/image', 'src', 'precomputed://' + args.src_path, (args.bbox_start[0]+args.bbox_stop[0])//2, (args.bbox_start[1]+args.bbox_stop[2])//2, args.bbox_start[0]))
  
  a = Aligner(args.model_path, args.max_displacement, edge_pad, mip_range, high_mip_chunk,
              args.src_path, args.tgt_path, args.dst_path, render_low_mip=render_mip, render_high_mip=max_mip,
              skip=args.skip, topskip=0, size=args.size, should_contrast=should_contrast,
              num_targets=num_targets, flip_average=not args.no_flip_average,
              write_intermediaries=args.write_intermediaries, 
              upsample_residuals=args.upsample_residuals, old_upsample=args.old_upsample, 
              old_vectors=args.old_vectors, ignore_field_init=args.ignore_field_init)
 
  # interleave coords by flattening
  coords = [x for t in zip(args.bbox_start[:2], args.bbox_stop[:2]) for x in t]
  bbox = BoundingBox(*coords, mip=0, max_mip=max_mip)
  if not a.check_all_params():
    raise Exception("Not all parameters are set")
  z_range = range(args.bbox_start[2], args.bbox_stop[2])
  multi_match(a, bbox, z_range) 
  cpc(a, bbox, z_range) 
