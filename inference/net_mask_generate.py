import sys
import torch
from args import get_argparser, parse_args, get_aligner, get_bbox
from os.path import join
import util

class NetMasker():
  def __init__(self, src_path, dst_path, mip, bbox_start, bbox_stop,
               bbox_mip, disable_cuda):
    bbox = Bbox(bbox_start, bbox_stop)
    self.device = None
    if not disable_cuda and torch.cuda.is_available():
      self.device = torch.device('cuda')
    else:
      self.device = torch.device('cpu')

    self.src = util.get_cloudvolume(src_path, mip=src_mip)
    self.src_bbox = self.src.bbox_to_mip(bbox, bbox_mip, mip)
    self.dst = util.create_cloudvolume(dst_path, self.src.info,
                                       mip, mip)

    self.archive = ###

  def run(self):
    z_range = range(self.src_bbox.minpt[2], self.src_bbox.maxpt[2])
    for z in z_range:
      print('Blurring z={0}'.format(z))
      self.src_bbox.minpt[2] = z
      self.src_bbox.maxpt[2] = z+1
      dst_bbox = self.src_bbox
      print('src_bbox {0}'.format(self.src_bbox))
      # print('dst_bbox {0}'.format(dst_bbox))
      src_img = util.get_image(self.src, self.src_bbox)
      mask = self.archive.model(src_img)
      util.save_image(self.dst, dst_bbox, mask)


if __name__ == '__main__':
  parser.add_argument('--mask_name',
    help='The name of the mask in NG', default='mask')
  args = parse_args(parser)

  args.tgt_path = join(args.dst_path, 'args.mask_name')
  # only compute matches to previous sections
  args.serial_operation = True
  m = get_masker(args)
  bbox = get_bbox(args)

  z_range = range(args.bbox_start[2], args.bbox_stop[2])
  a.dst[0].add_composed_cv(args.bbox_start[2], inverse=False)
  field_k = a.dst[0].get_composed_key(args.bbox_start[2], inverse=False)
  field_cv= a.dst[0].for_read(field_k)
  dst_cv = a.dst[0].for_write('dst_img')
  z_offset = 1
  uncomposed_field_cv = a.dst[z_offset].for_read('field')

  mip = args.mip
  composed_range = z_range[3:]
  if args.align_start:
    copy_range = z_range[0:1]
    uncomposed_range = z_range[1:3]
  else:
    copy_range = z_range[0:3]
    uncomposed_range = z_range[0:0]

  # copy first section
  for z in copy_range:
    print('Copying z={0}'.format(z))
    a.copy_section(z, dst_cv, z, bbox, mip)
    a.downsample(dst_cv, z, bbox, a.render_low_mip, a.render_high_mip)
  # align without vector voting
  for z in uncomposed_range:
    print('Aligning without vector voting z={0}'.format(z))
    src_z = z
    tgt_z = z-1
    a.compute_section_pair_residuals(src_z, tgt_z, bbox)
    a.render_section_all_mips(src_z, uncomposed_field_cv, src_z,
                              dst_cv, src_z, bbox, mip)
  # align with vector voting
  for z in composed_range:
    print('Aligning with vector voting z={0}'.format(z))
    a.generate_pairwise([z], bbox, render_match=False)
    a.compose_pairwise([z], args.bbox_start[2], bbox, mip,
                       forward_compose=True,
                       inverse_compose=False)
    a.render_section_all_mips(z, field_cv, z, dst_cv, z, bbox, mip)
