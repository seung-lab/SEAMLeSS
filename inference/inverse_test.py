import sys
import torch
from os.path import join
from args import get_argparser, parse_args, get_aligner, get_bbox 

def compose(a, f_cv, invf_cv, gw_cv, z, z_offset, bbox, mip):
    f = a.get_field(f_cv, z, bbox, mip, relative=True, to_tensor=True)
    invf = a.get_field(invf_cv, z-z_offset, bbox, mip, relative=True, to_tensor=True)
    # g = a.compose_fields(f, invf)
    g = a.compose_fields(invf, f)
    a.save_residual_patch(gw_cv, z, g, bbox, mip) 

def compose_chunkwise(a, f_cv, invf_cv, gw_cv, z, z_offset, bbox, mip):
  chunks = a.break_into_chunks(bbox, a.dst[0].dst_chunk_sizes[mip],
                                  a.dst[0].dst_voxel_offsets[mip], mip=mip, render=True)

  def chunkwise(patch_bbox):
    compose(a, f_cv, invf_cv, gw_cv, z, z_offset, patch_bbox, mip)
  a.pool.map(chunkwise, chunks)

if __name__ == '__main__':
  parser = get_argparser()
  parser.add_argument('--test_num', help='test number', type=int)
  args = parse_args(parser) 
  a = get_aligner(args)
  bbox = get_bbox(args)
  mip = args.mip

  z_range = range(args.bbox_start[2], args.bbox_stop[2])
  field_k = 'compose_test/field/{:01d}'.format(args.test_num)
  path = join(args.dst_path, field_k) 
  a.dst[0].add_path(field_k, path, data_type='float32', num_channels=2)
  a.dst[0].create_cv(field_k)
  z_offset = 1
  f_cv = a.dst[z_offset].for_read('field')
  invf_cv = a.dst[-z_offset].for_read('field')
  print('f_cv.path {0}'.format(f_cv.path))
  print('invf_cv.path {0}'.format(invf_cv.path))
  gw_cv = a.dst[0].for_write(field_k)
  gr_cv = a.dst[0].for_read(field_k)
  
  dst_k = 'compose_test/image/{:01d}'.format(args.test_num)
  path = join(args.dst_path, dst_k) 
  a.dst[0].add_path(dst_k, path, data_type='uint8', num_channels=1)
  a.dst[0].create_cv(dst_k)
  dst_cv = a.dst[0].for_write(dst_k)
  for z in z_range:
    compose_chunkwise(a, f_cv, invf_cv, gw_cv, z, z_offset, bbox, mip)
    a.render_section_all_mips(z, gr_cv, z, dst_cv, z, bbox, mip)


