from args import get_argparser, parse_args, get_aligner, get_bbox 

def render(aligner, bbox, z):
  aligner.total_bbox = bbox
  aligner.zs = z
  aligner.render_section_all_mips(z, bbox)

if __name__ == '__main__':
  parser = get_argparser()
  args = parse_args(parser) 
  a = get_aligner(args)
  bbox = get_bbox(args)

  for z in range(args.bbox_start[2], args.bbox_stop[2]): 
    print('Rendering z={0}'.format(z))
    render(a, bbox, z)

