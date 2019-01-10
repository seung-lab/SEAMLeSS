from args import get_argparser, parse_args, get_aligner, get_bbox 

if __name__ == '__main__':
  parser = get_argparser()
  parser.add_argument('--src_z', type=int, help='z of source image')
  parser.add_argument('--tgt_z', type=int, help='z of target image')
  args = parse_args(parser) 
  a = get_aligner(args)
  bbox = get_bbox(args)
  a.total_bbox = bbox
  a.zs = src_z
  a.compute_section_pair_residuals(src_z, tgt_z, bbox)
