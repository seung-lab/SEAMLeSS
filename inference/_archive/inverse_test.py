import sys
import torch
from os.path import join
from args import get_argparser, parse_args, get_aligner, get_bbox 

def compose(a, F_cv, invF_cv, gw_cv, z, bbox, mip):
  f = a.get_field(F_cv, z, bbox, mip, relative=True, to_tensor=True)
  invf = a.get_field(invF_cv, z, bbox, mip, relative=True, to_tensor=True)
  # g = a.compose_fields(f, invf)
  g = a.compose_fields(invf, f)
  a.save_residual_patch(gw_cv, z, g, bbox, mip) 

def compose_chunkwise(a, F_cv, invF_cv, gw_cv, z, bbox, mip):
  chunks = a.break_into_chunks(bbox, a.dst[0].dst_chunk_sizes[mip],
                                  a.dst[0].dst_voxel_offsets[mip], mip=mip, render=True)

  def chunkwise(patch_bbox):
    compose(a, F_cv, invF_cv, gw_cv, z, patch_bbox, mip)
  a.pool.map(chunkwise, chunks)

if __name__ == '__main__':
  parser = get_argparser()
  parser.add_argument('--test_num', help='test number', type=int)
  parser.add_argument('--block_size',
    help='batch size for regularization; batches are necessary to prevent large vectors from accumulating during composition',
    type=int, default=10) 
  args = parse_args(parser) 
  a = get_aligner(args)
  bbox = get_bbox(args)
  
  mip = args.mip
  overlap = args.tgt_radius
  if args.block_size < 2*overlap:
    args.block_size= 2*overlap 
  z_start = args.bbox_start[2]
  z_stop = args.bbox_stop[2]

  root = 'regularized/test'
  field_k = join(root, 'inverse_test_{:01d}/field'.format(args.test_num))
  path = join(args.dst_path, field_k) 
  a.dst[0].add_path(field_k, path, data_type='float32', num_channels=2)
  a.dst[0].create_cv(field_k)
  gw_cv = a.dst[0].for_write(field_k)
  gr_cv = a.dst[0].for_read(field_k)
  
  dst_k = join(root, 'inverse_test{:01d}/image'.format(args.test_num))
  path = join(args.dst_path, dst_k) 
  a.dst[0].add_path(dst_k, path, data_type='uint8', num_channels=1)
  a.dst[0].create_cv(dst_k)
  dst_cv = a.dst[0].for_write(dst_k)

  for block_start in range(z_start, z_stop, args.block_size - overlap):
    reg_range = range(block_start, block_start + args.block_size)
    # a.dst[0].add_composed_cv(block_start, inverse=False)
    # a.dst[0].add_composed_cv(block_start, inverse=True)
    a.dst[0].add_regularized_cv(block_start, inverse=False)
    a.dst[0].add_regularized_cv(block_start, inverse=True)
    F_cv = a.dst[0].get_regularized_cv(block_start, inverse=False, for_read=True)
    invF_cv = a.dst[0].get_regularized_cv(block_start, inverse=True, for_read=True)
    print('F_cv.path {0}'.format(F_cv.path))
    print('invF_cv.path {0}'.format(invF_cv.path))
    for z in reg_range:
      compose_chunkwise(a, F_cv, invF_cv, gw_cv, z, bbox, mip)
      a.render_section_all_mips(z, gr_cv, z, dst_cv, z, bbox, mip)

