import sys
import torch
from args import get_argparser, parse_args, get_aligner, get_bbox 

if __name__ == '__main__':
  parser = get_argparser()
  parser.add_argument('--tgt_path', type=str,
    help='IGNORED: tgt_path will be set to dst_path/image')
  args = parse_args(parser)
  args.tgt_path = join(args.dst_path, 'image')
  a = get_aligner(args)
  bbox = get_bbox(args)

  z_range = range(args.bbox_start[2], args.bbox_stop[2])
  a.dst[0].add_composed_cv(args.bbox_start[2], inverse=False)
  field_k = a.dst[0].get_composed_key(args.bbox_start[2], inverse=False)
  field_cv= a.dst[0].for_read(field_k)
  dst_cv = a.dst[0].for_write('dst_img')

  for z in z_range:
    a.generate_pairwise([z], bbox, render_match=False)
    a.compose_pairwise([z], args.bbox_start[2], bbox, mip,
                       forward_compose=True,
                       inverse_compose=False)
    a.render_section_all_mips(z, field_cv, z, dst_cv, z, bbox, mip)
