import sys
import torch
from args import get_argparser, parse_args, get_aligner, get_bbox 

if __name__ == '__main__':
  parser = get_argparser()
  parser.add_argument('--render_match', 
    help='render source with all single pairwise transforms before weighting',
    action='store_true')
  parser.add_argument('--forward_match', 
    help='generate matches for all pairs for z to z-i',
    action='store_true')
  parser.add_argument('--reverse_match', 
    help='generate matches for all pairs for z to z+i',
    action='store_true')
  parser.add_argument('--batch_size', type=int, default=10,
    help='if distributed, no. of sections to run simultaneously before proceeding')
  args = parse_args(parser) 
  a = get_aligner(args)
  bbox = get_bbox(args)

  z_range = range(args.bbox_start[2], args.bbox_stop[2])
  # multi_match(a, bbox, z_range) 
  a.generate_pairwise(z_range, bbox, forward_match=args.forward_match, 
                      reverse_match=args.reverse_match, 
                      render_match=args.render_match, batch_size=args.batch_size)

