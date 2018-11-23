from process import Process
from mipless_cloudvolume import MiplessCloudVolume as CV 
from cloudvolume.lib import Vec
import torch
from torch.nn.functional import interpolate
import numpy as np
import os
from os.path import join
import json
import math
from time import time
from copy import deepcopy, copy
import scipy
import scipy.ndimage
from normalizer import Normalizer
from vector_vote import vector_vote, get_diffs, weight_diffs, \
                        compile_field_weights, weighted_sum_fields
from temporal_regularization import create_field_bump
from helpers import save_chunk, crop, upsample, gridsample_residual, np_downsample

from skimage.morphology import disk as skdisk
from skimage.filters.rank import maximum as skmaximum

from boundingbox import BoundingBox

from pathos.multiprocessing import ProcessPool, ThreadPool
from threading import Lock

import torch.nn as nn

class SrcDir():
  def __init__(self, src_path, tgt_path, 
                     src_mask_path, tgt_mask_path, 
                     src_mask_mip, tgt_mask_mip,
                     src_mask_val, tgt_mask_val):
    self.vols = {}
    self.kwargs = {'bounded': False, 'fill_missing': True, 'progress': False}
    self.vols['src_img'] = CV(src_path, **self.kwargs) 
    self.vols['tgt_img'] = CV(tgt_path, **self.kwargs) 
    if src_mask_path:
      self.read['src_mask'] = CV(src_mask_path, **self.read_kwargs) 
    if tgt_mask_path:
      self.read['tgt_mask'] = CV(tgt_mask_path, **self.read_kwargs) 
    self.src_mask_mip = src_mask_mip
    self.tgt_mask_mip = tgt_mask_mip
    self.src_mask_val = src_mask_val
    self.tgt_mask_val = tgt_mask_val

  def __getitem__(self, k):
    return self.vols[k]

  def __contains__(self, k):
    return k in self.vols

class DstDir():
  """Manager of CloudVolumes required by the Aligner
  
  Manage CloudVolumes used for reading & CloudVolumes used for writing. Read & write
  distinguished by the different sets of kwargs that are used for the CloudVolume.
  All CloudVolumes are MiplessCloudVolumes. 
  """
  def __init__(self, dst_path, src_cv, mip_range, provenance, max_offset):
    print('Creating DstDir for {0}'.format(dst_path))
    self.mip_range = mip_range
    self.root = dst_path
    self.paths = {}
    self.dst_chunk_sizes   = []
    self.dst_voxel_offsets = []
    self.vec_chunk_sizes   = []
    self.vec_voxel_offsets = []
    self.vec_total_sizes   = []
    self.read = {}
    self.write = {}
    self.read_kwargs = {'bounded': False, 'fill_missing': True, 'progress': False}
    self.write_kwargs = {'bounded': False, 'fill_missing': True, 'progress': False, 
                  'autocrop': True, 'non_aligned_writes': False}
    self.info = None
    self.provenance = provenance
    self.create_info(src_cv, max_offset)
    self.add_default_cv()
  
  def for_read(self, k):
    return self.read[k]

  def for_write(self, k):
    return self.write[k]
  
  def __getitem__(self, k):
    return self.read[k]

  def __contains__(self, k):
    return k in self.read

  def create_info(self, src_cv, max_offset):
    src_info = src_cv.info
    m = len(src_info['scales'])
    each_factor = Vec(2,2,1)
    factor = Vec(2**m,2**m,1)
    for _ in self.mip_range: 
      src_cv.add_scale(factor)
      factor *= each_factor
      chunksize = src_info['scales'][-2]['chunk_sizes'][0] // each_factor
      src_info['scales'][-1]['chunk_sizes'] = [ list(map(int, chunksize)) ]

    self.info = deepcopy(src_info)
    chunk_size = self.info["scales"][0]["chunk_sizes"][0][0]
    dst_size_increase = max_offset
    if dst_size_increase % chunk_size != 0:
      dst_size_increase = dst_size_increase - (dst_size_increase % max_offset) + chunk_size
    scales = self.info["scales"]
    for i in range(len(scales)):
      scales[i]["voxel_offset"][0] -= int(dst_size_increase / (2**i))
      scales[i]["voxel_offset"][1] -= int(dst_size_increase / (2**i))

      scales[i]["size"][0] += int(dst_size_increase / (2**i))
      scales[i]["size"][1] += int(dst_size_increase / (2**i))

      x_remainder = scales[i]["size"][0] % scales[i]["chunk_sizes"][0][0]
      y_remainder = scales[i]["size"][1] % scales[i]["chunk_sizes"][0][1]

      x_delta = 0
      y_delta = 0
      if x_remainder != 0:
        x_delta = scales[i]["chunk_sizes"][0][0] - x_remainder
      if y_remainder != 0:
        y_delta = scales[i]["chunk_sizes"][0][1] - y_remainder

      scales[i]["size"][0] += x_delta
      scales[i]["size"][1] += y_delta

      scales[i]["size"][0] += int(dst_size_increase / (2**i))
      scales[i]["size"][1] += int(dst_size_increase / (2**i))
      #make it slice-by-slice writable
      scales[i]["chunk_sizes"][0][2] = 1

      self.dst_chunk_sizes.append(scales[i]["chunk_sizes"][0][0:2])
      self.dst_voxel_offsets.append(scales[i]["voxel_offset"])
    
    for i in range(len(scales)):
      self.vec_chunk_sizes.append(scales[i]["chunk_sizes"][0][0:2])
      self.vec_voxel_offsets.append(scales[i]["voxel_offset"])
      self.vec_total_sizes.append(scales[i]["size"])

  def create_cv(self, name, data_type, num_channels):
    path = self.paths[name]
    provenance = self.provenance 
    info = deepcopy(self.info)
    info['data_type'] = data_type
    info['num_channels'] = num_channels
    self.read[name] = CV(path, info=info, provenance=provenance, **self.read_kwargs)
    self.write[name] = CV(path, info=info, provenance=provenance, **self.write_kwargs)

  def add_path(self, k, path):
    self.paths[k] = path

  def add_default_cv(self):
    root = self.root
    self.add_path('dst_img', join(root, 'image'))
    self.add_path('field', join(root, 'field'))
    self.create_cv('dst_img', 'uint8', 1)
    self.create_cv('field', 'float32', 2)
  
  def get_composed_key(self, compose_start, inverse):
    return '{0}{1}'.format('invF' if inverse else 'F', compose_start)
  
  def add_composed_cv(self, compose_start, inverse):
    """Create CloudVolume for storing composed vector fields

    Args
       compose_start: int, indicating the earliest section used for composing
       inverse: bool indicating whether composition aligns COMPOSE_START to Z (True),
        or Z to COMPOSE_START (False)
    """
    k = self.get_composed_key(compose_start, inverse)
    path = join(self.root, 'composed', self.get_composed_key(compose_start, inverse))
    self.add_path(k, path)
    self.create_cv(k, 'float32', 2)

class Aligner:
  """
  Destination directory structure
  * z_3:  pairwise fields that align z to z-3 and write to z
  * z_2  
  * z_1
  * z_1i: pairwise fields that align z to z+1 and write to z
  * z_2i
  * z_3i
  * vector_vote
    * F_START:    composed fields from vector voting that align to START
    * Fi_START:   composed inverse fields from vector voting that align to START
  * reg_field: the final, regularized field
  """
  def __init__(self, model_path, max_displacement, crop,
               mip_range, high_mip_chunk, src_path, tgt_path, dst_path, 
               src_mask_path='', src_mask_mip=0, src_mask_val=1, 
               tgt_mask_path='', tgt_mask_mip=0, tgt_mask_val=1,
               align_across_z=1, disable_cuda=False, max_mip=12,
               render_low_mip=2, render_high_mip=6, is_Xmas=False, threads=5,
               max_chunk=(1024, 1024), max_render_chunk=(2048*2, 2048*2),
               skip=0, topskip=0, size=7, should_contrast=True, 
               disable_flip_average=False, write_intermediaries=False,
               upsample_residuals=False, old_upsample=False, old_vectors=False,
               ignore_field_init=False, z=0, tgt_radius=1, **kwargs):
    self.process_high_mip = mip_range[1]
    self.process_low_mip  = mip_range[0]
    self.render_low_mip   = render_low_mip
    self.render_high_mip  = render_high_mip
    self.high_mip         = max(self.render_high_mip, self.process_high_mip)
    self.high_mip_chunk   = high_mip_chunk
    self.max_chunk        = max_chunk
    self.max_render_chunk = max_render_chunk
    self.max_mip          = max_mip
    self.size = size
    self.old_vectors = old_vectors
    self.ignore_field_init = ignore_field_init
    self.write_intermediaries = write_intermediaries

    self.max_displacement = max_displacement
    self.crop_amount      = crop
    self.disable_cuda = disable_cuda
    self.device = torch.device('cpu') if disable_cuda else torch.device('cuda')
    
    provenance = {}
    provenance['project'] = 'seamless'
    provenance['src_path'] = src_path
    provenance['tgt_path'] = tgt_path

    self.src = SrcDir(src_path, tgt_path, 
                      src_mask_path, tgt_mask_path, 
                      src_mask_mip, tgt_mask_mip, 
                      src_mask_val, tgt_mask_val)
    src_cv = self.src['src_img'][0]
    self.dst = {}
    self.tgt_range = range(-tgt_radius, tgt_radius+1)
    for i in self.tgt_range:
      if i > 0:
        path = '{0}/z_{1}'.format(dst_path, abs(i))
      elif i < 0:
        path = '{0}/z_{1}i'.format(dst_path, abs(i))
      else: 
        path = dst_path
      self.dst[i] = DstDir(path, src_cv, mip_range, provenance, max_displacement)

    self.net = Process(model_path, mip_range[0], is_Xmas=is_Xmas, cuda=True, 
                       dim=high_mip_chunk[0]+crop*2, skip=skip, 
                       topskip=topskip, size=size, 
                       flip_average=not disable_flip_average, old_upsample=old_upsample)

    self.normalizer = Normalizer(min(5, mip_range[0])) 
    self.upsample_residuals = upsample_residuals
    self.pool = ThreadPool(threads)

  def Gaussian_filter(self, kernel_size, sigma):
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)
    channels =1
    mean = (kernel_size - 1)/2.
    variance = sigma**2.
    gaussian_kernel = (1./(2.*math.pi*variance)) *np.exp(
        -torch.sum((xy_grid - mean)**2., dim=-1) /\
        (2*variance))
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,kernel_size=kernel_size, padding = (kernel_size -1)//2, bias=False)
    gaussian_filter.weight.data = gaussian_kernel.type(torch.float32)
    gaussian_filter.weight.requires_grad = False
    return gaussian_filter

  def set_chunk_size(self, chunk_size):
    self.high_mip_chunk = chunk_size

  def get_upchunked_bbox(self, bbox, chunk_size, offset, mip):
    raw_x_range = bbox.x_range(mip=mip)
    raw_y_range = bbox.y_range(mip=mip)

    x_chunk = chunk_size[0]
    y_chunk = chunk_size[1]

    x_offset = offset[0]
    y_offset = offset[1]

    x_remainder = ((raw_x_range[0] - x_offset) % x_chunk)
    y_remainder = ((raw_y_range[0] - y_offset) % y_chunk)

    x_delta = 0
    y_delta = 0
    if x_remainder != 0:
      x_delta =  x_chunk - x_remainder
    if y_remainder != 0:
      y_delta =  y_chunk - y_remainder

    calign_x_range = [raw_x_range[0] + x_delta, raw_x_range[1]]
    calign_y_range = [raw_y_range[0] + y_delta, raw_y_range[1]]

    x_start = calign_x_range[0] - x_chunk
    y_start = calign_y_range[0] - y_chunk

    x_start_m0 = x_start * 2**mip
    y_start_m0 = y_start * 2**mip

    result = BoundingBox(x_start_m0, x_start_m0 + bbox.x_size(mip=0),
                         y_start_m0, y_start_m0 + bbox.y_size(mip=0),
                         mip=0, max_mip=self.max_mip) #self.process_high_mip)
    return result

  def compose_fields(self, f, g):
    """Compose two fields f & g, for f(g(x))
    """    
    g = g.permute(0,3,1,2)
    return f + gridsample_residual(g, f, padding_mode='border').permute(0,2,3,1)
    
  def get_composed_field(self, src_z, tgt_z, compose_start, bbox, mip, 
                                inverse=False, relative=False, to_tensor=True):
    """Get composed field for Z_LIST using CloudVolume dirs at Z_OFFSET_LIST. Use field
    in BBOX at MIP. Use INVERSE to left-compose the next field in the list. Use RELATIVE
    to return a vector field in range [-1,1], and use TO_TENSOR to return a Tensor object.
    """
    z_offset = src_z - tgt_z
    f_cv = self.dst[z_offset].get_read('field')
    composed_k = self.dst[0].get_composed_key(compose_start, inverse)
    F_cv = self.dst[0].for_read(composed_k)
    f = self.get_field(f_cv, z, bbox, mip, relative=True, to_tensor=to_tensor)
    F = self.get_field(F_cv, z, bbox, mip, relative=True, to_tensor=to_tensor)
    if inverse:
      f, F = F, f
    F = self.compose_fields(F, f)

    if not relative:
      F = self.rel_to_abs_residual(F, mip)
    return F 

  def blur_field(self, field, std=128):
    """Apply Gaussian with std to a vector field
    """
    print('blur_field')
    regular_part_x = torch.from_numpy(scipy.ndimage.filters.gaussian_filter((field[...,0]), std)).unsqueeze(-1)
    regular_part_y = torch.from_numpy(scipy.ndimage.filters.gaussian_filter((field[...,1]), std)).unsqueeze(-1)
    #regular_part = self.gauss_filter(field.permute(3,0,1,2))
    #regular_part = torch.from_numpy(self.reg_field) 
    #field = decay_factor * field + (1 - decay_factor) * regular_part.permute(1,2,3,0) 
    #field = regular_part.permute(1,2,3,0) 
    field = torch.cat([regular_part_x,regular_part_y],-1)
    return field.to(device=self.device)

  def get_field(self, cv, z, bbox, mip, relative=False, to_tensor=True):
    x_range = bbox.x_range(mip=mip)
    y_range = bbox.y_range(mip=mip)
    print('get_field from {0}, MIP{1} @ z={2}'.format(cv.path, mip, z))
    field = cv[mip][x_range[0]:x_range[1], y_range[0]:y_range[1], z]
    res = np.expand_dims(np.squeeze(field), axis=0)
    if relative:
      res = self.abs_to_rel_residual(res, bbox, mip)
    if to_tensor:
      res = torch.from_numpy(res)
      return res.to(device=self.device)
    else:
      return res

  def save_vector_patch(self, cv, z, field, bbox, mip):
    field = field.data.cpu().numpy() 
    x_range = bbox.x_range(mip=mip)
    y_range = bbox.y_range(mip=mip)
    field = np.squeeze(field)[:, :, np.newaxis, :]
    print('save_vector_patch from {0}, MIP{1} at z={2}'.format(cv.path, mip, z))
    cv[mip][x_range[0]:x_range[1], y_range[0]:y_range[1], z] = field

  def save_residual_patch(self, cv, z, res, crop, bbox, mip):
    print ("Saving residual patch {} at MIP {}".format(bbox.__str__(mip=0), mip))
    v = res * (res.shape[-2] / 2) * (2**mip)
    v = v[:,crop:-crop, crop:-crop,:]
    self.save_vector_patch(cv, z, v, bbox, mip)

  def break_into_chunks(self, bbox, chunk_size, offset, mip, render=False):
    chunks = []
    raw_x_range = bbox.x_range(mip=mip)
    raw_y_range = bbox.y_range(mip=mip)

    x_chunk = chunk_size[0]
    y_chunk = chunk_size[1]

    x_offset = offset[0]
    y_offset = offset[1]

    x_remainder = ((raw_x_range[0] - x_offset) % x_chunk)
    y_remainder = ((raw_y_range[0] - y_offset) % y_chunk)

    x_delta = 0
    y_delta = 0
    if x_remainder != 0:
      x_delta =  x_chunk - x_remainder
    if y_remainder != 0:
      y_delta =  y_chunk - y_remainder

    calign_x_range = [raw_x_range[0] - x_remainder, raw_x_range[1]]
    calign_y_range = [raw_y_range[0] - y_remainder, raw_y_range[1]]

    x_start = calign_x_range[0] - x_chunk
    y_start = calign_y_range[0] - y_chunk

    if (self.process_high_mip > mip):
        high_mip_scale = 2**(self.process_high_mip - mip)
    else:
        high_mip_scale = 1

    processing_chunk = (int(self.high_mip_chunk[0] * high_mip_scale),
                        int(self.high_mip_chunk[1] * high_mip_scale))
    if not render and (processing_chunk[0] > self.max_chunk[0]
                      or processing_chunk[1] > self.max_chunk[1]):
      processing_chunk = self.max_chunk
    elif render and (processing_chunk[0] > self.max_render_chunk[0]
                     or processing_chunk[1] > self.max_render_chunk[1]):
      processing_chunk = self.max_render_chunk

    for xs in range(calign_x_range[0], calign_x_range[1], processing_chunk[0]):
      for ys in range(calign_y_range[0], calign_y_range[1], processing_chunk[1]):
        chunks.append(BoundingBox(xs, xs + processing_chunk[0],
                                 ys, ys + processing_chunk[0],
                                 mip=mip, max_mip=self.max_mip)) #self.high_mip))

    return chunks

  def vector_vote(self, z, compose_start, bbox, mip, inverse, T=1):
    """Compute consensus vector field using pairwise vector fields with earlier sections. 

    Vector voting requires that vector fields be composed to a common section
    before comparison: inverse=False means that the comparison will be based on 
    composed vector fields F_{z,compose_start}, while inverse=True will be
    F_{compose_start,z}.

    Args:
       z: int, section whose pairwise vector fields will be used
       compose_start: int, the first pairwise vector field to use in calculating
         any composed vector fields
       bbox: BoundingBox, the region of interest over which to vote
       mip: int, the data MIP level
       inverse: bool, indicates the direction of composition to use 
    """
    fields = []
    prior_z = [i for i in self.tgt_range if i < 0]
    for z_offset in prior_z: 
      src_z = z
      tgt_z = src_z + z_offset
      if inverse:
        src_z, tgt_z = tgt_z, src_z 
      F = self.get_composed_field(src_z, tgt_z, compose_start, bbox, mip, 
                                  inverse=inverse, relative=False, to_tensor=True)

    field = vector_vote(fields, T=T)
    composed_k = self.dst[0].get_composed_key(compose_start, inverse)
    F_cv = self.dst[0].for_write(composed_k)
    self.save_vector_patch(F_cv, z, field, bbox, mip)

    # if self.write_intermediaries:
    #   self.save_image_patch('diffs', diffs.cpu().numpy(), bbox, mip, to_uint8=False)
    #   self.save_image_patch('diff_weights', diffs.cpu().numpy(), bbox, mip, to_uint8=False)
    #   self.save_image_patch('weights', diffs.cpu().numpy(), bbox, mip, to_uint8=False)

  def compute_residual_patch(self, src_z, tgt_z, out_patch_bbox, mip):
    """Predict vector field that will warp section at SOURCE_Z to section at TARGET_Z
    within OUT_PATCH_BBOX at MIP. Vector field will be stored at SOURCE_Z, using DST at
    SOURCE_Z - TARGET_Z. 

    Args
      src_z: int of section to be warped
      tgt_z: int of section to be warped to
      out_patch_bbox: BoundingBox for region of both sections to process
      mip: int of MIP level to use for OUT_PATCH_BBOX 
    """
    assert(src_z != tgt_z)
    print ("Computing residual for region {}.".format(out_patch_bbox.__str__(mip=0)), flush=True)
    precrop_patch_bbox = deepcopy(out_patch_bbox)
    precrop_patch_bbox.uncrop(self.crop_amount, mip=mip)

    src_cv = self.src['src_img']
    tgt_cv = self.src['tgt_img']
    src_patch = self.get_image(src_cv, src_z, precrop_patch_bbox, mip,
                                adjust_contrast=True, to_tensor=True)
    tgt_patch = self.get_image(tgt_cv, tgt_z, precrop_patch_bbox, mip,
                                adjust_contrast=True, to_tensor=True) 

    if 'src_mask' in self.src:
      mask_cv = self.src['src_mask']
      src_mask = self.get_mask(mask_cv, src_z, precrop_patch_bbox, 
                           src_mip=self.src.src_mask_mip,
                           dst_mip=mip, valid_val=self.src.src_mask_val)
      src_patch = src_patch.masked_fill_(src_mask, 0)
    if 'tgt_mask' in self.src:
      mask_cv = self.src['tgt_mask']
      tgt_mask = self.get_mask(mask_cv, tgt_z, precrop_patch_bbox, 
                           src_mip=self.src.tgt_mask_mip,
                           dst_mip=mip, valid_val=self.src.tgt_mask_val)
      tgt_patch = tgt_patch.masked_fill_(tgt_mask, 0)
    X = self.net.process(src_patch, tgt_patch, mip, crop=self.crop_amount, 
                                                 old_vectors=self.old_vectors)
    field, residuals, encodings, cum_residuals = X

    # save the final vector field for warping
    z_offset = src_z - tgt_z
    field_cv = self.dst[z_offset].for_write('field')
    self.save_vector_patch(field_cv, src_z, field, out_patch_bbox, mip)

    # if self.write_intermediaries:
    #   mip_range = range(self.process_low_mip+self.size-1, self.process_low_mip-1, -1)
    #   for res_mip, res, cumres in zip(mip_range, residuals[1:], cum_residuals[1:]):
    #       crop = self.crop_amount // 2**(res_mip - self.process_low_mip)   
    #       self.save_residual_patch('res', src_z, src_z_offset, res, crop, out_patch_bbox, res_mip)
    #       self.save_residual_patch('cumres', src_z, src_z_offset, cumres, crop, out_patch_bbox, res_mip)
    #       if self.upsample_residuals:
    #         crop = self.crop_amount   
    #         res = self.scale_residuals(res, res_mip, self.process_low_mip)
    #         self.save_residual_patch('resup', src_z, z_offset, res, crop, out_patch_bbox, 
    #                                  self.process_low_mip)
    #         cumres = self.scale_residuals(cumres, res_mip, self.process_low_mip)
    #         self.save_residual_patch('cumresup', src_z, z_offset, cumres, crop, 
    #                                  out_patch_bbox, self.process_low_mip)

    #   print('encoding size: {0}'.format(len(encodings)))
    #   for k, enc in enumerate(encodings):
    #       mip = self.process_low_mip + k
    #       # print('encoding shape @ idx={0}, mip={1}: {2}'.format(k, mip, enc.shape))
    #       crop = self.crop_amount // 2**k
    #       enc = enc[:,:,crop:-crop, crop:-crop].permute(2,3,0,1)
    #       enc = enc.data.cpu().numpy()
    #       
    #       def write_encodings(j_slice, z):
    #         x_range = out_patch_bbox.x_range(mip=mip)
    #         y_range = out_patch_bbox.y_range(mip=mip)
    #         patch = enc[:, :, :, j_slice]
    #         # uint_patch = (np.multiply(patch, 255)).astype(np.uint8)
    #         cv(self.paths['enc'][mip], 
    #             mip=mip, bounded=False, 
    #             fill_missing=True, autocrop=True, 
    #             progress=False, provenance={})[x_range[0]:x_range[1],
    #                             y_range[0]:y_range[1], z, j_slice] = patch 
  
    #       # src_image encodings
    #       write_encodings(slice(0, enc.shape[-1] // 2), src_z)
    #       # dst_image_encodings
    #       write_encodings(slice(enc.shape[-1] // 2, enc.shape[-1]), tgt_z)

  def rel_to_abs_residual(self, field, mip):    
    """Convert vector field from relative space [-1,1] to absolute space
    """
    return field * (field.shape[-2] / 2) * (2**mip)

  def abs_to_rel_residual(self, abs_residual, patch, mip):
    """Convert vector field from absolute space to relative space [-1,1]
    """
    x_fraction = patch.x_size(mip=0) * 0.5
    y_fraction = patch.y_size(mip=0) * 0.5

    rel_residual = deepcopy(abs_residual)
    rel_residual[0, :, :, 0] /= x_fraction
    rel_residual[0, :, :, 1] /= y_fraction
    return rel_residual

  def get_bbox_id(self, in_bbox, mip):
    raw_x_range = self.total_bbox.x_range(mip=mip)
    raw_y_range = self.total_bbox.y_range(mip=mip)

    x_chunk = self.dst[0].dst_chunk_sizes[mip][0]
    y_chunk = self.dst[0].dst_chunk_sizes[mip][1]

    x_offset = self.dst[0].dst_voxel_offsets[mip][0]
    y_offset = self.dst[0].dst_voxel_offsets[mip][1]

    x_remainder = ((raw_x_range[0] - x_offset) % x_chunk)
    y_remainder = ((raw_y_range[0] - y_offset) % y_chunk)
     
    calign_x_range = [raw_x_range[0] - x_remainder, raw_x_range[1]]
    calign_y_range = [raw_y_range[0] - y_remainder, raw_y_range[1]]

    calign_x_len = raw_x_range[1] - raw_x_range[0] + x_remainder
    #calign_y_len = raw_y_range[1] - raw_y_range[0] + y_remainder

    in_x_range = in_bbox.x_range(mip=mip)
    in_y_range = in_bbox.y_range(mip=mip)
    in_x_len = in_x_range[1] - in_x_range[0]
    in_y_len = in_y_range[1] - in_y_range[0]
    line_bbox_num = (calign_x_len + in_x_len -1)// in_x_len
    cid = ((in_y_range[0] - calign_y_range[0]) // in_y_len) * line_bbox_num + (in_x_range[0] - calign_x_range[0]) // in_x_len
    return cid

  ## Patch manipulation
  def warp_patch(self, src_z, field_cv, field_z, bbox, mip):
    """Non-chunk warping

    From BBOX at MIP, warp image at SRC_Z in CloudVolume SRC_CV using
    field at FIELD_Z in CloudVolume FIELD_CV.
    """
    influence_bbox = deepcopy(bbox)
    influence_bbox.uncrop(self.max_displacement, mip=0)
    start = time()
    
    field = self.get_field(field_cv, field_z, influence_bbox, mip, 
                           relative=True, to_tensor=True)
    mip_disp = int(self.max_displacement / 2**mip)
    src_cv = self.src['src_img']
    image = self.get_image(src_cv, src_z, influence_bbox, mip, 
                           adjust_contrast=False, to_tensor=True)
    if 'src_mask' in self.src:
      mask_cv = self.src['src_mask']
      mask = self.get_mask(mask_cv, src_z, influence_bbox, 
                           src_mip=self.src.src_mask_mip,
                           dst_mip=mip, valid_val=self.src.src_mask_val)
      image = image.masked_fill_(mask, 0)

    # print('warp_patch shape {0}'.format(image.shape))
    # no need to warp if flow is identity since warp introduces noise
    if torch.min(field) != 0 or torch.max(field) != 0:
      image = gridsample_residual(image, field, padding_mode='zeros')
    else:
      print ("not warping")
    # print('warp_image image1.shape: {0}'.format(image.shape))
    if self.disable_cuda:
      image = image.numpy()[:,:,mip_disp:-mip_disp,mip_disp:-mip_disp]
    else:
      image = image.cpu().numpy()[:,:,mip_disp:-mip_disp,mip_disp:-mip_disp]
    # print('warp_image image3.shape: {0}'.format(image.shape))
    return image

  def downsample_patch(self, cv, z, bbox, mip):
    data = self.get_image(cv, z, bbox, mip, adjust_contrast=False, to_tensor=True)
    data = interpolate(data, scale_factor=0.5, mode='bilinear')
    return data.cpu().numpy()

  ## Data saving
  def save_image_patch(self, cv, z, float_patch, bbox, mip, to_uint8=True):
    x_range = bbox.x_range(mip=mip)
    y_range = bbox.y_range(mip=mip)
    patch = np.transpose(float_patch, (3,2,1,0))
    if to_uint8:
      patch = (np.multiply(patch, 255)).astype(np.uint8)
    cv[mip][x_range[0]:x_range[1], y_range[0]:y_range[1], z] = patch

  def scale_residuals(self, res, src_mip, dst_mip):
    print('Upsampling residuals from MIP {0} to {1}'.format(src_mip, dst_mip))
    up = nn.Upsample(scale_factor=2, mode='bilinear')
    for m in range(src_mip, dst_mip, -1):
      res = up(res.permute(0,3,1,2)).permute(0,2,3,1)
    return res

  ## Data loading
  def dilate_mask(self, mask, radius=5):
    return skmaximum(np.squeeze(mask).astype(np.uint8), skdisk(radius)).reshape(mask.shape).astype(np.bool)
    
  def get_mask(self, cv, z, bbox, src_mip, dst_mip, valid_val, to_tensor=True):
    data = self.get_data(cv, z, bbox, src_mip=src_mip, dst_mip=dst_mip, 
                             to_float=False, adjust_contrast=False, 
                             to_tensor=to_tensor)
    return data == valid_val

  def get_image(self, cv, z, bbox, mip, adjust_contrast=False, to_tensor=True):
    return self.get_data(cv, z, bbox, src_mip=mip, dst_mip=mip, to_float=True, 
                             adjust_contrast=adjust_contrast, to_tensor=to_tensor)

  def get_data(self, cv, z, bbox, src_mip, dst_mip, to_float=True, 
                     adjust_contrast=False, to_tensor=True):
    """Retrieve CloudVolume data. Returns 4D ndarray or tensor, BxCxWxH
    
    Args:
       cv_key: string to lookup CloudVolume
       bbox: BoundingBox defining data range
       src_mip: mip of the CloudVolume data
       dst_mip: mip of the output mask (dictates whether to up/downsample)
       to_float: output should be float32
       adjust_contrast: output will be normalized
       to_tensor: output will be torch.tensor
    """
    x_range = bbox.x_range(mip=src_mip)
    y_range = bbox.y_range(mip=src_mip)
    data = cv[src_mip][x_range[0]:x_range[1], y_range[0]:y_range[1], z] 
    data = np.transpose(data, (3,2,1,0))
    if to_float:
      data = np.divide(data, float(255.0), dtype=np.float32)
    if adjust_contrast:
      data = self.normalizer.apply(data).reshape(data.shape)
    # convert to tensor if requested, or if up/downsampling required
    if to_tensor | (src_mip != dst_mip):
      data = torch.from_numpy(data).to(device=self.device)
      if src_mip != dst_mip:
        # k = 2**(src_mip - dst_mip)
        size = (bbox.y_size(dst_mip), bbox.x_size(dst_mip))
        if not isinstance(data, torch.cuda.ByteTensor): #TODO: handle device
          data = interpolate(data, size=size, mode='bilinear')
        else:
          data = data.type('torch.cuda.DoubleTensor')
          data = interpolate(data, size=size, mode='nearest')
          data = data.type('torch.cuda.ByteTensor')
      if not to_tensor:
        data = data.cpu().numpy()
    
    return data

  ## High level services
  def copy_section(self, z, dst_cv, dst_z, bbox, mip):
    print ("moving section {} mip {} to dest".format(z, mip), end='', flush=True)
    start = time()
    chunks = self.break_into_chunks(bbox, self.dst[0].dst_chunk_sizes[mip],
                                    self.dst[0].dst_voxel_offsets[mip], mip=mip, render=True)
    #for patch_bbox in chunks:
    def chunkwise(patch_bbox):

      src_cv = self.src['src_img']
      if 'src_mask' in self.src:
        mask_cv = self.src['src_mask']
        raw_patch = self.get_image(src_cv, z, patch_bbox, mip,
                                    adjust_contrast=False, to_tensor=True)
        raw_mask = self.get_mask(mask_cv, z, precrop_patch_bbox, 
                                 src_mip=self.src.src_mask_mip,
                                 dst_mip=mip, valid_val=self.src.src_mask_val)
        raw_patch = raw_patch.masked_fill_(raw_mask, 0)
        raw_patch = raw_patch.cpu().numpy()
      else: 
        raw_patch = self.get_image(src_cv, z, patch_bbox, mip,
                                    adjust_contrast=False, to_tensor=False)
      self.save_image_patch(dst_cv, dst_z, raw_patch, patch_bbox, mip)

    self.pool.map(chunkwise, chunks)

    end = time()
    print (": {} sec".format(end - start))

  def render(self, src_z, field_cv, field_z, dst_cv, dst_z, bbox, mip):
    """Chunkwise render

    Warp the image in BBOX at MIP and SRC_Z in CloudVolume dir at SRC_Z_OFFSET, 
    using the field at FIELD_Z in CloudVolume dir at FIELD_Z_OFFSET, and write 
    the result to DST_Z in CloudVolume dir at DST_Z_OFFSET. Chunk BBOX 
    appropriately.
    """
    print('Rendering src_z={0} from z_offset={1} @ MIP{2} to dst_z={3} at z_offset={4}'.format(src_z, src_z_offset, mip, dst_z, dst_z_offset), flush=True)
    start = time()
    chunks = self.break_into_chunks(bbox, self.dst[0].dst_chunk_sizes[mip],
                                    self.dst[0].dst_voxel_offsets[mip], mip=mip, render=True)

    def chunkwise(patch_bbox):
      warped_patch = self.warp_patch(src_z, field_cv, field_z, patch_bbox, mip)
      # print('warp_image render.shape: {0}'.format(warped_patch.shape))
      self.save_image_patch(dst_cv, dst_z, warped_patch, patch_bbox, mip)
    self.pool.map(chunkwise, chunks)
    end = time()
    print (": {} sec".format(end - start))

  def downsample(self, cv, z, bbox, source_mip, target_mip):
    """Chunkwise downsample

    For the CloudVolume dirs at Z_OFFSET, warp the SRC_IMG using the FIELD for
    section Z in region BBOX at MIP. Chunk BBOX appropriately and save the result
    to DST_IMG.
    """
    print ("Downsampling {} from mip {} to mip {}".format(bbox.__str__(mip=0), source_mip, target_mip))
    for m in range(source_mip+1, target_mip + 1):
      chunks = self.break_into_chunks(bbox, self.dst[0].dst_chunk_sizes[m],
                                      self.dst[0].dst_voxel_offsets[m], mip=m, render=True)

      def chunkwise(patch_bbox):
        print ("Downsampling {} to mip {}".format(patch_bbox.__str__(mip=0), m))
        downsampled_patch = self.downsample_patch(cv, z, patch_bbox, m-1)
        self.save_image_patch(cv, z, downsampled_patch, patch_bbox, m)
      self.pool.map(chunkwise, chunks)

  def render_section_all_mips(self, src_z, field_cv, field_z, dst_cv, dst_z, bbox, mip):
    self.render(src_z, field_cv, field_z, dst_cv, dst_z, bbox, self.render_low_mip)
    self.downsample(dst_cv, dst_z, bbox, self.render_low_mip, self.render_high_mip)

  def compute_section_pair_residuals(self, src_z, tgt_z, bbox):
    """Chunkwise vector field inference for section pair

    For the CloudVolume dirs at Z_OFFSET, warp the SRC_IMG using the FIELD for
    section Z in region BBOX at MIP. Chunk BBOX appropriately and save the result
    to DST_IMG.
    """
    for m in range(self.process_high_mip,  self.process_low_mip - 1, -1):
      start = time()
      chunks = self.break_into_chunks(bbox, self.dst[0].vec_chunk_sizes[m],
                                      self.dst[0].vec_voxel_offsets[m], mip=m)
      print ("Aligning slice {} to slice {} at mip {} ({} chunks)".
             format(src_z, tgt_z, m, len(chunks)), flush=True)

      #for patch_bbox in chunks:
      def chunkwise(patch_bbox):
      #FIXME Torch runs out of memory
      #FIXME batchify download and upload
        self.compute_residual_patch(src_z, tgt_z, patch_bbox, mip=m)
      self.pool.map(chunkwise, chunks)
      end = time()
      print (": {} sec".format(end - start))

      # If m > self.process_low_mip:
      #     self.prepare_source(src_z, bbox, m - 1)
    
  def count_box(self, bbox, mip):    
    chunks = self.break_into_chunks(bbox, self.dst[0].dst_chunk_sizes[mip],
                                      self.dst[0].dst_voxel_offsets[mip], mip=mip, render=True)
    total_chunks = len(chunks)
    self.image_pixels_sum =np.zeros(total_chunks)
    self.field_sf_sum =np.zeros((total_chunks, 2), dtype=np.float32)

  def vector_vote_chunkwise(self, z, compose_start, bbox, mip, inverse, T=-1):
    """Chunked-processing of vector voting
    
    Args:
       z: section of fields to weight
       compose_start: int of earliest section to use in composition
       bbox: boundingbox of region to process
       mip: field MIP level
       T: softmin temperature (default will be 2**mip)
    """
    start = time()
    chunks = self.break_into_chunks(bbox, self.dst[0].vec_chunk_sizes[mip],
                                    self.dst[0].vec_voxel_offsets[mip], mip=mip)
    print("Vector voting for slice {0} @ MIP{1} ({2} chunks)".
           format(z, mip, len(chunks)), flush=True)

    #for patch_bbox in chunks:
    def chunkwise(patch_bbox):
      self.vector_vote(z, compose_start, patch_bbox, mip, inverse=inverse, T=T)
    self.pool.map(chunkwise, chunks)
    end = time()
    print (": {} sec".format(end - start))

  def multi_match(self, z, render=True):
    """Match Z to all sections within TGT_RADIUS

    Args:
       render: bool indicating whether to render section
    """
    bbox = self.total_bbox
    mip = self.process_low_mip
    for z_offset in self.tgt_range:
      if z_offset != 0: 
        src_z = z
        tgt_z = src_z + z_offset
        self.compute_section_pair_residuals(src_z, tgt_z, bbox)
        if render:
          field_cv = self.dst[z_offset].for_read('field')
          dst_cv = self.dst[z_offset].for_read('dst_img')
          dst_z = src_z
          if z_offset > 0:
            dst_z = tgt_z
          self.render_section_all_mips(src_z, field_cv, src_z, dst_cv, src_z, bbox, mip)

  def generate_pairwise(self, z_range, bbox, render_match=False):
    """Create all pairwise matches for each SRC_Z in Z_RANGE to each TGT_Z in TGT_RADIUS
  
    Args:
        z_range: list of z indices to be matches 
        bbox: BoundingBox object for bounds of 2D region
        render_match: bool indicating whether to separately render out
            each aligned section before compiling vector fields with voting
            (useful for debugging)
    """
    self.total_bbox = bbox
    mip = self.process_low_mip
    for z in z_range:
      self.multi_match(z, render=render_match)
 
  def compose_pairwise(self, z_range, compose_start, bbox, mip, inverse=False, both=False):
    """Combine pairwise vector fields in TGT_RADIUS using vector voting, while composing
    with earliest section at COMPOSE_START.

    Args
       z_range: list of ints (assumed to be monotonic & sequential)
       compose_start: int of earliest section used in composition
       bbox: BoundingBox defining chunk region
       mip: int for MIP level of data
       inverse: bool, indicating whether to compose with inverse or forward transforms
       both: bool, indicating whether to compose with both inverse and forward transforms
        (ignores INVERSE)
    """
    self.total_bbox = bbox
    T = 2**mip
    print('softmin temp: {0}'.format(T))
    self.dst[0].add_composed_cv(compose_start, inverse=inverse)
    if both: 
      self.dst[0].add_composed_cv(compose_start, inverse=not inverse)
    for z in z_range:
      self.vector_vote(z, compose_start, bbox, mip, inverse=inverse, T=T)
      if both:
        self.vector_vote(z, compose_start, bbox, mip, inverse=not inverse, T=T)

  def get_composed_neighborhood(self, z, compose_start, bbox, inverse=True):
    """Compile all composed vector fields that warp neighborhood in TGT_RANGE to Z

    Args
       z: int for index of SRC section
       compose_start: int of earliest section used in composition
       bbox: BoundingBox defining chunk region
       mip: int for MIP level of data
    """
    fields = []
    Fk = self.dst[0].get_composed_key(compose_start, inverse=inverse)
    F_cv = self.dst[0].for_read(Fk)
    for z_offset in self.tgt_range:
      tgt_z = z + z_offset
      F = self.get_field(F_cv, tgt_z, bbox, mip, relative=False, to_tensor=to_tensor)
      fields.append(F)
    return torch.cat(fields, 0)
 
  def shift_composed_neighborhood(self, Fs, z, bbox, mip, inverse=True):
    """Shift composed neighborhood by dropping earliest z & appending next z
  
    Args
       invFs: 4D torch tensor of inverse composed vector vote fields
       z: int representing the z of the input invFs. invFs will be shifted to z+1.
       bbox: BoundingBox representing xy extent of invFs
       mip: int for data resolution of the field
    """
    Fk = self.dst[0].get_composed_key(compose_start, inverse=False)
    F_cv = self.dst[0].for_read(Fk)
    next_z = z + self.tgt_range[-1] + 1
    next_F = self.get_field(F_cv, tgt_z, bbox, mip, relative=True, to_tensor=True)
    return torch.cat((Fs[1:, ...], nextF), 0)

  def regularize_z(self, z_range, compose_start, bbox, mip, sigma=1.4):
    """For a given chunk, temporally regularize each Z in Z_RANGE
    
    Make Z_RANGE as large as possible to avoid IO: self.shift_field
    is called to add and remove the newest and oldest sections.

    Args
       z_range: list of ints (assumed to be monotonic & sequential)
       compose_start: int of earliest section used in composition
       bbox: BoundingBox defining chunk region
       mip: int for MIP level of data
    """
    z = z_range[0]
    Fk = self.dst[0].get_composed_key(compose_start, inverse=False)
    F_cv = self.dst[0].for_read(Fk)
    regF_cv = self.dst[0].for_write('field')
    invFs = self.get_composed_neighborhood(z, compose_start, bbox)
    bump_dims = invFs.shape 
    bump = create_field_bump(bump_dims, sigma)

    for z in z_range:
      invF_avg = torch.sum(torch.mul(bump, field), dim=0, keepdim=True)
      F = self.get_field(F_cv, z, relative=True, to_tensor=True)
      regF = self.compose(invF_avg, F)
      self.save_residual_patch(regF_cv, z, regF, bbox, mip)
      if z != z_range[-1]:
        invFs = self.shift_composed_neighborhood(invFs, z, bbox, mip)

  def regularize_z_chunkwise(self, z_range, compose_start, bbox, mip, T):
    """Chunked-processing of temporal regularization 
    
    Args:
       z_range: int list, range of sections over which to regularize 
       bbox: BoundingBox of region to process
       mip: field MIP level
       T: softmin temperature 
    """
    start = time()
    chunks = self.break_into_chunks(bbox, self.dst[0].vec_chunk_sizes[mip],
                                    self.dst[0].vec_voxel_offsets[mip], mip=mip)
    print("Regularizing slice range {0} @ MIP{1} ({2} chunks)".
           format(z_range, mip, len(chunks)), flush=True)

    #for patch_bbox in chunks:
    def chunkwise(patch_bbox):
      self.regularize_z(z_range, compose_start, patch_bbox, mip, T=T)
    self.pool.map(chunkwise, chunks)
    end = time()
    print (": {} sec".format(end - start))
  
