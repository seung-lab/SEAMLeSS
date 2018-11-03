"""
Vector voting with pairwise vector fields

* Vector fields that warp z -> z+k are stored at z
* There is only one vector field_sf stored during vector_voting
* get_field has a flag to compose with the field_sf 
* There is an update field_sf method
* Weights for vector votes are calculated using vector fields
  each composed with its field_sf
* The weighted sum field includes the field_sf
* 
"""

import sys
import torch
from args import get_argparser, parse_args, get_aligner, get_bbox 

def align_vector_vote(aligner, bbox, z_range, start_without=True, 
                                        render_multi_match=False):
  """Align stack of images using vector voting on previous 3 slices

  Args:
      aligner: Aligner object
      bbox: BoundingBox object for bounds of 2D region
      z_range: list of z indices to align
      start_without: Bool indicating whether to align 3 slices without
          vector voting
      render_multi_match: Bool indicating whether to separately render out
          each aligned section before compiling vector fields with voting
          (useful for debugging)
  """
  aligner.total_bbox = bbox
  start_z = z_range[0]
  if start_without:
    aligner.align_ng_stack(z_range[0], z_range[2], bbox, move_anchor=False) 
    z_range = z_range[3:]
  for z in z_range:
    other_zs = [z-1, z-2, z-3]
    field_paths = aligner.multi_match(z, other_zs, inverse=False, 
                                                   render=render_multi_match)
    # print('align_vector_vote field_paths: {0}'.format(field_paths))
    mip = aligner.process_low_mip
    T = 2**mip
    print('softmin temp: {0}'.format(T))
    aligner.reset()
    aligner.compute_weighted_field(field_paths, z, bbox, mip, T)
    aligner.render_section_all_mips(z, bbox, start_z)

def multi_match(aligner, bbox, z_range, inverse=True):
  """Match all pairs of sections in z range

  Args:
      aligner: Aligner object
      bbox: BoundingBox object for bounds of 2D region
      z_range: list of z indices to align
      inverse: Bool indicating whether to treat each z as src or tgt
          e.g. For vector voting, inverse should be set to False so that each
          z is used as the src in each section pair. For generating a mask by
          comparing multiple alignments, inverse should be to True, so that
          z is used as the tgt in each section pair (alignments will all be
          in the coordinate frame of z).
  """
  aligner.total_bbox = bbox
  for z in z_range:
    other_zs = list(set(z_range) - set([z]))
    aligner.multi_match(z, other_zs, inverse=inverse, render=True)

if __name__ == '__main__':
  parser = get_argparser()
  args = parse_args(parser) 
  a = get_aligner(args)
  bbox = get_bbox(args)

  z_range = range(args.bbox_start[2], args.bbox_stop[2])
  # multi_match(a, bbox, z_range) 
  align_vector_vote(a, bbox, z_range, start_without=False)
