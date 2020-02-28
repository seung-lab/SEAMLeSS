import torch
from cloudvolume import CloudVolume
from cloudvolume.lib import Bbox, Vec, find_closest_divisor
from taskqueue import RegisteredTask, TaskQueue, LocalTaskQueue, GreenTaskQueue
from tasks import FailedInvertTask
import numpy as np
from copy import copy
import torchfields
import sys

import gevent.monkey
gevent.monkey.patch_all(thread=False)

from os.path import join

def get_cloudvolume(path, mip):
  return CloudVolume(path, mip=mip, fill_missing=True)

def get_image(vol, bbox):
  return vol[bbox.to_slices()]

def save_image(dst, bbox, img):
  dst[bbox.to_slices()] = img

def to_float(img):
  return img.astype(np.float)

def to_tensor(img, device=torch.device('cpu')):
  img = torch.from_numpy(img)
  return img.permute(2,3,1,0).to(device=device, non_blocking=True)

def to_numpy(ten):
  if ten.is_cuda:
    img = ten.permute(3,2,0,1).cpu().numpy()
  else:
    img = ten.permute(3,2,0,1).numpy()
  return img
  
def diff_to_numpy(ten):
  if ten.is_cuda:
    img = ten.permute(2,3,1,0).cpu().numpy()
  else:
    img = ten.permute(2,3,1,0).numpy()
  return img

def R_to_uint8(img):
  return to_uint8(((img + 1) / 2)*255)

def uint8_to_R(img):
  return to_float((img / 255.) * 2 - 1)

def to_uint8(img):
  return img.astype(np.uint8)

def get_composite_image(vol, bbox, reverse=True):
  """Collapse 3D image into a 2D image, replacing black pixels in the first 
      z slice with the nearest nonzero pixel in other slices.
  
  Args:
  * vol: CloudVolume object
  * bbox: CloudVolume Bbox object
  * reverse: bool indicating to start with the last section (highest z),
      then fill in missing data with earlier secitons.
  """
  img = get_image(vol, bbox)
  if reverse:
    o = img[:,:,-1:,:]
    for z in range(img.shape[2]-1,0,-1):
      o[o <= 1] = img[:,:,z-1:z,:][o <= 1]
  else:
    o = img[:,:,:1,:]
    for z in range(1, img.shape[2]):
      o[o <= 1] = img[:,:,z:z+1,:][o <= 1]
  return o

def create_cloudvolume(dst_path, info, src_mip, dst_mip):
  dst_info = copy(info)
  dst_info['scales'] = dst_info['scales'][:1]
  dst_info['scales'][0]['encoding'] = 'raw'
  chunksize = dst_info['scales'][0]['chunk_sizes'][0]
  chunksize[2] = 1
  dst_info['scales'][0]['chunk_sizes'][0] = chunksize
  each_factor = Vec(2,2,1)
  factor = each_factor.clone()
  for m in range(1, dst_mip+1):
    add_scale(factor, dst_info)
    factor *= each_factor
  # dst_info['data_type'] = 'uint8'
  dst = CloudVolume(dst_path, mip=dst_mip, info=dst_info, 
             fill_missing=True, non_aligned_writes=True, cdn_cache=False)
  dst.commit_info()
  return dst

def get_field(cv, bbox, device=torch.device('cpu')):
  f = cv[bbox.to_slices()]
  return torch.from_numpy(f).permute(2,0,1,3).to(device=device)

def get_inverse_field(cv, bbox, device=torch.device('cpu')):
  bbsize = bbox.size()
  t = get_field(cv, bbox, device=device) 

  # permute into (N, C, H, W) convention for displace fields
  tf = t.permute(0,3,1,2).float()

  # normalize displacement fields
  tf[:, 0, :, :].div_(bbsize[0] / 2)
  tf[:, 1, :, :].div_(bbsize[1] / 2) 
  
  # cast to field and invert each z index
  f = tf.field()
  g = f.clone()
  for z in range(0, f.size()[0]):
    g[z, :, :, :] = ~f[z, :, :, :] 

  # unnormalize, cast back to tensor, unpermute, and match type
  g[:, 0, :, :].mul_(bbsize[0] / 2)
  g[:, 1, :, :].mul_(bbsize[1] / 2)

  g = g.tensor().permute(0, 2, 3, 1).type(t.dtype)
  return g


def field_to_numpy(field):
  if field.is_cuda:
    field = field.data.cpu().numpy() 
  else:
    field = field.data.numpy()
  return field 
  
def save_field(cv, bbox, field):
  cv[bbox.to_slices()] = np.transpose(field, (1,2,0,3))

def add_scale(factor, info):
  """
  Generate a new downsample scale to for the info file and return an updated dictionary.
  You'll still need to call self.commit_info() to make it permenant.
  Required:
    factor: int (x,y,z), e.g. (2,2,1) would represent a reduction of 2x in x and y
  Returns: info dict
  """
  # e.g. {"encoding": "raw", "chunk_sizes": [[64, 64, 64]], "key": "4_4_40", 
  # "resolution": [4, 4, 40], "voxel_offset": [0, 0, 0], 
  # "size": [2048, 2048, 256]}
  fullres = info['scales'][0]

  # If the voxel_offset is not divisible by the ratio,
  # zooming out will slightly shift the data.
  # Imagine the offset is 10
  #    the mip 1 will have an offset of 5
  #    the mip 2 will have an offset of 2 instead of 2.5 
  #        meaning that it will be half a pixel to the left
  
  chunk_size = find_closest_divisor(fullres['chunk_sizes'][0], closest_to=[64,64,64])

  def downscale(size, roundingfn):
    smaller = Vec(*size, dtype=np.float32) / Vec(*factor)
    return list(map(int, roundingfn(smaller)))

  newscale = {
    u"encoding": fullres['encoding'],
    u"chunk_sizes": [ list(map(int, chunk_size)) ],
    u"resolution": list(map(int, Vec(*fullres['resolution']) * factor )),
    u"voxel_offset": downscale(fullres['voxel_offset'], np.floor),
    u"size": downscale(fullres['size'], np.ceil),
  }

  if newscale['encoding'] == 'compressed_segmentation':
    newscale['compressed_segmentation_block_size'] = fullres['compressed_segmentation_block_size']

  newscale[u'key'] = str("_".join([ str(res) for res in newscale['resolution']]))

  new_res = np.array(newscale['resolution'], dtype=int)

  preexisting = False
  for index, scale in enumerate(info['scales']):
    res = np.array(scale['resolution'], dtype=int)
    if np.array_equal(new_res, res):
      preexisting = True
      info['scales'][index] = newscale
      break

  if not preexisting:    
    info['scales'].append(newscale)

  return newscale 

def write_failed_queue(queue_name, file_name):
    f = open(file_name, "a")
    orig_stdout = sys.stdout
    sys.stdout = f
    with GreenTaskQueue(queue_name=queue_name) as tq:
      tq.poll(lease_seconds=10)
    f.close()
    f = sys.stdout


