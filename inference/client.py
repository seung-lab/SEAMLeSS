from args import get_argparser, parse_args, get_aligner, get_bbox 

if __name__ == '__main__':
  parser = get_argparser()
  parser.add_argument('--move_anchor', 
    help='copy first section of run from src to dst', 
    action='store_true')
  args = parse_args(parser) 
  a = get_aligner(args)
  bbox = get_bbox(args)
  a.align_ng_stack(args.bbox_start[2], args.bbox_stop[2], bbox, move_anchor=args.move_anchor)
