import sys
import torch
from args import get_argparser, parse_args, get_aligner, get_bbox 

if __name__ == '__main__':
  parser = get_argparser()
  parser.add_argument('--compose_start', help='earliest section composed', type=int)
  args = parse_args(parser) 
  a = get_aligner(args)
  bbox = get_bbox(args)
  args.serial_operation = False

  mip = args.mip
  z_range = range(args.bbox_start[2], args.bbox_stop[2])
  print('Composing for z_range {0}'.format(z_range))
  a.compose_pairwise(z_range, args.compose_start, bbox, mip,
                     forward_compose=True, inverse_compose=False)
