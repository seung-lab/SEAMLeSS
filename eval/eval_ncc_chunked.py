from cloudvolume import CloudVolume
from cloudvolume.lib import Bbox, Vec
import numpy as np
from copy import copy

import argparse

def get_image(vol, origin, block):
  bbox = Bbox(origin, origin+block)
  return vol[bbox.to_slices()]

def normalize(img):
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

def grid_score(cv, bbox, block):
  """Return image of score run at grid within bbox
  """
  src_img = cv[bbox.to_slices()][:,:,:,0]
  dst_img = cv[(bbox+Vec(0,0,1)).to_slices()][:,:,:,0]
  intervals = bbox.size3() // block
  o = np.zeros(intervals)
  n = np.prod(intervals)
  for x in range(intervals[0]):
    for y in range(intervals[1]):
      for z in range(intervals[2]):
        print('{0} / {1}\r'.format((x+1)*(y+1)*(z+1), n), end='')
        src_origin = Vec(x,y,z)*block
        src = get_image(src_img, src_origin, block)
        dst = get_image(dst_img, src_origin, block)
        o[x,y,z] = score(src, dst)
  return o

def main(src_path, src_mip, dst_path, dst_mip, bbox, bbox_mip):
  """Run similarity score in a grid over bbox, compile as image & save. 
  """
  src = CloudVolume(src_path, mip=src_mip, fill_missing=True)
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

  sds = 2**(dst_mip - src_mip)
  dst_chunk = Vec(sds, sds, 1)
  ssb = sdb = 2**(src_mip - bbox_mip)
  src_bbox = bbox // Vec(ssb, ssb, 1)
  src_bbox = src_bbox.shrink_to_chunk_size(dst_chunk)
  # sdb = 2**(dst_mip - bbox_mip)
  dst_bbox = src_bbox // Vec(sds, sds, 1)
  o = grid_score(src, src_bbox, dst_chunk)
  dst[dst_bbox.to_slices()] = to_uint8(o)[:,:,:,np.newaxis]

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  paraser.add_argument('--src_path', type=str)
  paraser.add_argument('--dst_path', type=str)
  paraser.add_argument('--src_mip', type=int)
  paraser.add_argument('--dst_mip', type=int)
  paraser.add_argument('--x_start', type=int)
  paraser.add_argument('--x_stop', type=int)
  paraser.add_argument('--y_start', type=int)
  paraser.add_argument('--y_stop', type=int)
  paraser.add_argument('--z_start', type=int)
  paraser.add_argument('--z_stop', type=int)
  paraser.add_argument('--bbox_mip', type=int)
  args = parser.parse_args()

  bbox = Bbox([args.x_start, args.y_start, args.z_start],
                [args.x_stop, args.y_stop, args.z_stop])
  main(args.src_path, args.src_mip, args.dst_path, args.dst_mip, 
                                                bbox, args.bbox_mip)

# test
# src_path = 'gs://neuroglancer/seamless/cprod_smooth4_mip8_full/image'
# dst_path = 'gs://neuroglancer/seamless/cprod_smooth4_mip8_full/image/qc_chunked'
# bbox = Bbox([185836, 149331, 1368], [217877, 188617, 1369])
# src_mip = 6
# dst_mip = 12
# bbox_mip = 0
# main(src_path, src_mip, dst_path, dst_mip, bbox, bbox_mip)