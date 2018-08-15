from cloudvolume import CloudVolume
from cloudvolume.lib import Bbox, Vec
import numpy as np
from copy import copy
from PIL import Image

import argparse

def get_bbox(origin, block):
  return Bbox(origin, origin+block)

def get_image(vol, bbox):
  return vol[bbox.to_slices()]

def get_composite_image(vol, bbox):
  """Collapse 3D image into a 2D image, replacing black pixels in the first 
      z slice with the nearest nonzero pixel in other slices.
  """
  img = vol[bbox.to_slices()]
  o = img[:,:,0]
  for z in range(1, img.shape[2]):
    o[o <= 1] = img[:,:,z][o <= 1]
  return o

def normalize(img):
  if np.std(img) == 0:
    return img
  return (img - np.mean(img)) / np.std(img)

def score(src_img, dst_img):
  s = normalize(src_img.flatten())
  d = normalize(dst_img.flatten())
  r = np.correlate(s, d)[0] / src_img.size
  if np.isnan(r):
    return 0.0 
  else:
    return r

def to_uint8(s):
  return (((s + 1) / 2)*255).astype(np.uint8)

def grid_score(src_img, dst_img, block):
  """Return image of score run at grid of blocks between src_img & dst_img
  """
  intervals = src_img.shape // block
  o = np.zeros(intervals)
  n = np.prod(intervals)
  i = 0
  for x in range(intervals[0]):
    for y in range(intervals[1]):
      for z in range(intervals[2]):
        i += 1
        print('{1} : {0}\r'.format(i, n), end='')
        src_origin = Vec(x,y,z)*block
        src = get_image(src_img, get_bbox(src_origin, block))
        dst = get_image(dst_img, get_bbox(src_origin, block))
        o[x,y,z] = score(src, dst)
  return o

def main(src_path, src_mip, dst_path, dst_mips, bbox, bbox_mip, composite_z):
  """Run similarity score in a grid over bbox, compile as image & save. 

  Args:
    * src_path: path to CloudVolume with images to be scored
    * src_mip: MIP level of images to be used in scoring
    * dst_path: path to CloudVolume where score image to be written
    * dst_mips: MIP levels of output to be used. This will dictate the blocksize
        used in scoring: 2**(dst_mip - src_mip). 
        Requires min(dst_mips) >= src_mip.
    * bbox: Requested bbox of area to be scored.
    * bbox_mip: MIP level of the bbox (typically 0)
    * composite_z: The number of slices to use when making a composite
        image to be scored against.

  Outputs:
    * Writes a CloudVolume at dst_path

  """
  assert(min(dst_mips) >= src_mip)
  src = CloudVolume(src_path, mip=src_mip, fill_missing=True, parallel=2)

  for z in range(bbox.minpt[2], bbox.maxpt[2]):
    print('Scoring z={0}'.format(z))
    bbox.minpt[2] = z
    bbox.maxpt[2] = z+1

    max_dst_mip = max(dst_mips)
    sds = 2**(max_dst_mip - src_mip)
    max_dst_chunk = Vec(sds, sds, 1)
    ssb = sdb = 2**(src_mip - bbox_mip)
    src_bbox = bbox // Vec(ssb, ssb, 1)
    src_bbox = src_bbox.shrink_to_chunk_size(max_dst_chunk)
    src_img = get_image(src, src_bbox)[:,:,:,0]
    composite_bbox = Bbox(src_bbox.minpt+Vec(0,0,1), 
                          src_bbox.maxpt+Vec(0,0,composite_z+1))
    dst_img = get_composite_image(src, composite_bbox)

    for dst_mip in dst_mips:
      print('Compiling @ dst_mip={0}'.format(dst_mip))
      sds = 2**(dst_mip - src_mip)
      dst_chunk = Vec(sds, sds, 1)
      dst_bbox = src_bbox // Vec(sds, sds, 1)
      o = grid_score(src_img, dst_img, dst_chunk)

      dst_info = copy(src.info)
      dst_info['scales'] = dst_info['scales'][0:1]
      each_factor = Vec(2,2,1)
      factor = each_factor.clone()
      for m in range(1, dst_mip+1):
        src.add_scale(factor, info=dst_info)
        factor *= each_factor
      dst_info['data_type'] = 'uint8'
      dst = CloudVolume(dst_path, mip=dst_mip, info=dst_info, 
                    fill_missing=True, non_aligned_writes=True, cdn_cache=False)
      dst.commit_info()
      dst[dst_bbox.to_slices()] = to_uint8(o)[:,:,:,np.newaxis]

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Create score image.')
  parser.add_argument('--src_path', type=str, 
    help='Path to CloudVolume with images to be scored')
  parser.add_argument('--dst_path', type=str,
    help='Path to CloudVolume where score image to be written')
  parser.add_argument('--src_mip', type=int,
    help='MIP level of images to be used in scoring')
  parser.add_argument('--dst_mips', nargs='+', type=int,
    help='MIP levels of output to be written. This dictates chunksize used.')
  parser.add_argument('--bbox_start', nargs=3, type=int,
    help='bbox origin, 3-element int list')
  parser.add_argument('--bbox_stop', nargs=3, type=int,
    help='bbox origin+shape, 3-element int list')
  parser.add_argument('--bbox_mip', type=int, default=0,
    help='MIP level at which bbox_start & bbox_stop are specified')
  parser.add_argument('--composite_z', type=int, default=0,
    help='Number of z slices to create a composite image for scoring')
  args = parser.parse_args()

  bbox = Bbox(args.bbox_start, args.bbox_stop)
  main(args.src_path, args.src_mip, args.dst_path, args.dst_mips, 
                                  bbox, args.bbox_mip, args.composite_z)

# test
# src_path = 'gs://neuroglancer/seamless/cprod_smooth4_mip8_full/image'
# dst_path = 'gs://neuroglancer/seamless/cprod_smooth4_mip8_full/image/qc_chunked'
# bbox = Bbox([185836, 149331, 1368], [217877, 188617, 1369])
# src_mip = 6
# dst_mip = 12
# bbox_mip = 0
# main(src_path, src_mip, dst_path, dst_mip, bbox, bbox_mip)