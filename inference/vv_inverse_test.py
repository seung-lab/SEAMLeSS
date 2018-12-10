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
  parser.add_argument('--compose_block',
    help='block size before using a new compose_start',
    type=int, default=0) 
  args = parse_args(parser) 
  a = get_aligner(args)
  bbox = get_bbox(args)
  
  mip = args.mip
  block_size = args.compose_block // 2
  # if args.compose_block == 0:
    # args.compose_block = args.bbox_stop[2] - args.bbox_start[2]
    # block_size = args.compose_block

  field_k = 'composed/test/field/vv_inverse_test_{:01d}'.format(args.test_num)
  path = join(args.dst_path, field_k) 
  a.dst[0].add_path(field_k, path, data_type='float32', num_channels=2)
  a.dst[0].create_cv(field_k)
  gw_cv = a.dst[0].for_write(field_k)
  gr_cv = a.dst[0].for_read(field_k)
  
  dst_k = 'composed/test/image/vv_inverse_test{:01d}'.format(args.test_num)
  path = join(args.dst_path, dst_k) 
  a.dst[0].add_path(dst_k, path, data_type='uint8', num_channels=1)
  a.dst[0].create_cv(dst_k)
  dst_cv = a.dst[0].for_write(dst_k)

  for block_start in range(args.bbox_start[2], args.bbox_stop[2], args.compose_block):
    z_range = range(block_start + block_size, block_start + block_size + args.compose_block)
    a.dst[0].add_composed_cv(block_start, inverse=False)
    a.dst[0].add_composed_cv(block_start, inverse=True)
    Fk = a.dst[0].get_composed_key(block_start, inverse=False)
    invFk = a.dst[0].get_composed_key(block_start, inverse=True)
    F_cv = a.dst[0].for_read(Fk)
    invF_cv = a.dst[0].for_read(invFk)
    print('F_cv.path {0}'.format(F_cv.path))
    print('invF_cv.path {0}'.format(invF_cv.path))
    for z in z_range:
      compose_chunkwise(a, F_cv, invF_cv, gw_cv, z, bbox, mip)
      a.render_section_all_mips(z, gr_cv, z, dst_cv, z, bbox, mip)


