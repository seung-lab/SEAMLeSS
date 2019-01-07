import sys
import torch
from args import get_argparser, parse_args, get_aligner, get_bbox
import argparse
import numpy as np

from os.path import join
import util
from pathlib import Path
from cloudvolume.lib import Bbox
from utilities.archive import ModelArchive

class NetMasker():
  def __init__(self, src_path, dst_path, model_path, compute_mip, bbox_start, bbox_stop,
               bbox_mip, disable_cuda):
    self.device = None
    if not disable_cuda and torch.cuda.is_available():
      self.device = torch.device('cuda')
    else:
      self.device = torch.device('cpu')

    self.src = util.get_cloudvolume(src_path, mip=compute_mip)
    bbox = Bbox(bbox_start, bbox_stop)
    self.src_bbox = self.src.bbox_to_mip(bbox, bbox_mip, compute_mip)
    self.dst = util.create_cloudvolume(dst_path, self.src.info,
                                       compute_mip, compute_mip)

    model_name = Path(model_path).stem
    self.archive = ModelArchive(model_name)

  def run(self):
    z_range = range(self.src_bbox.minpt[2], self.src_bbox.maxpt[2])
    for z in z_range:
      print('Computing mask for z={0}'.format(z))
      self.src_bbox.minpt[2] = z
      self.src_bbox.maxpt[2] = z+1
      dst_bbox = self.src_bbox

      src_img = util.get_image(self.src, self.src_bbox)[:, :, 0, 0]
      bool_mask = self.archive.model(src_img)
      util.save_image(self.dst, dst_bbox, bool_mask.astype(np.uint8) * 250)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--src_path', type=str,
    help='CloudVolume path')
  parser.add_argument('--dst_path', type=str,
    help='CloudVolume path')
  parser.add_argument('--compute_mip', type=int)
  parser.add_argument('--model_path', type=str, help='path to model')
  parser.add_argument('--bbox_start', nargs=3, type=int)
  parser.add_argument('--bbox_stop', nargs=3, type=int)
  parser.add_argument('--bbox_mip', type=int, help='MIP of the bbox start and stop')
  parser.add_argument('--disable_cuda', action='store_true', help='Disable CDUA')
  args = parser.parse_args()

  m = NetMasker(**vars(args))
  m.run()

