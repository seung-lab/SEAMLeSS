import sys
import torch
from args import get_argparser, parse_args, get_aligner, get_bbox 

if __name__ == '__main__':
  parser = get_argparser()
  parser.add_argument('--render_match', 
    help='render source with all single pairwise transforms before weighting',
    action='store_true')
  parser.add_argument('--render_final', 
    help='render final image with weighted field', 
    action='store_true')
  args = parse_args(parser) 
  a = get_aligner(args)
  bbox = get_bbox(args)

  z_range = range(args.bbox_start[2], args.bbox_stop[2])
  # multi_match(a, bbox, z_range) 
  a.align_stack_vector_vote(z_range, bbox,  
                             render_match=args.render_match,
                             render_final=args.render_final)
