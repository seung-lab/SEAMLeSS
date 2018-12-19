from process import Process
from mipless_cloudvolume import MiplessCloudVolume as CV 
from mipless_cloudvolume import deserialize_miplessCV as DCV
from cloudvolume.lib import Vec
import torch
from torch.nn.functional import interpolate
import numpy as np
import os
from os.path import join
import json
import math
from time import time, sleep
from copy import deepcopy, copy
import scipy
import scipy.ndimage
from normalizer import Normalizer
from vector_vote import vector_vote, get_diffs, weight_diffs, \
                        compile_field_weights, weighted_sum_fields
from temporal_regularization import create_field_bump
from utilities.helpers import save_chunk, crop, upsample, gridsample_residual, np_downsample
from helpers import  invert

from skimage.morphology import disk as skdisk
from skimage.filters.rank import maximum as skmaximum 
from boundingbox import BoundingBox, deserialize_bbox

from pathos.multiprocessing import ProcessPool, ThreadPool
from threading import Lock

import torch.nn as nn

from task_handler import TaskHandler, make_residual_task_message, \
        make_render_task_message, make_copy_task_message, \
        make_downsample_task_message, make_compose_task_message, \
        make_prepare_task_message, make_vector_vote_task_message, \
        make_regularize_task_message, make_render_low_mip_task_message

class SrcDir():
  def __init__(self, src_path, tgt_path, 
                     src_mask_path, tgt_mask_path, 
                     src_mask_mip, tgt_mask_mip,
                     src_mask_val, tgt_mask_val):
    self.vols = {}
    self.kwargs = {'bounded': False, 'fill_missing': True, 'progress': False}
    self.vols['src_img'] = CV(src_path, mkdir=False, **self.kwargs) 
    self.vols['tgt_img'] = CV(tgt_path, mkdir=False, **self.kwargs) 
    if src_mask_path:
      self.read['src_mask'] = CV(src_mask_path, mkdir=False, **self.read_kwargs) 
    if tgt_mask_path:
      self.read['tgt_mask'] = CV(tgt_mask_path, mkdir=False, **self.read_kwargs) 
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
  def __init__(self, dst_path, info, provenance, suffix=''):
    print('Creating DstDir for {0}'.format(dst_path))
    self.root = dst_path
    self.info = info
    self.provenance = provenance
    self.paths = {} 
    self.dst_chunk_sizes = []
    self.dst_voxel_offsets = []
    self.vec_chunk_sizes = [] 
    self.vec_voxel_offsets = []
    self.vec_total_sizes = []
    self.compile_scales()
    self.read = {}
    self.write = {}
    self.read_kwargs = {'bounded': False, 'fill_missing': True, 'progress': False}
    self.write_kwargs = {'bounded': False, 'fill_missing': True, 'progress': False, 
                  'autocrop': True, 'non_aligned_writes': False, 'cdn_cache': False}
    self.add_path('dst_img', join(self.root, 'image'), data_type='uint8', num_channels=1)
    self.add_path('dst_img_1', join(self.root, 'image1'), data_type='uint8', num_channels=1)
    self.add_path('field', join(self.root, 'field'), data_type='float32', num_channels=2)
    self.suffix = suffix
    self.create_paths()
  
  def for_read(self, k):
    return self.read[k]

  def for_write(self, k):
    return self.write[k]
  
  def __getitem__(self, k):
    return self.read[k]

  def __contains__(self, k):
    return k in self.read

  @classmethod
  def create_info(cls, src_cv, mip_range, max_offset):
    src_info = src_cv.info
    m = len(src_info['scales'])
    each_factor = Vec(2,2,1)
    factor = Vec(2**m,2**m,1)
    for _ in mip_range: 
      src_cv.add_scale(factor)
      factor *= each_factor
      chunksize = src_info['scales'][-2]['chunk_sizes'][0] // each_factor
      src_info['scales'][-1]['chunk_sizes'] = [ list(map(int, chunksize)) ]

    info = deepcopy(src_info)
    chunk_size = info["scales"][0]["chunk_sizes"][0][0]
    dst_size_increase = max_offset
    if dst_size_increase % chunk_size != 0:
      dst_size_increase = dst_size_increase - (dst_size_increase % max_offset) + chunk_size
    scales = info["scales"]
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
    return info

  def compile_scales(self):
    scales = self.info["scales"]
    for i in range(len(scales)):
      self.dst_chunk_sizes.append(scales[i]["chunk_sizes"][0][0:2])
      self.dst_voxel_offsets.append(scales[i]["voxel_offset"]) 
      self.vec_chunk_sizes.append(scales[i]["chunk_sizes"][0][0:2])
      self.vec_voxel_offsets.append(scales[i]["voxel_offset"])
      self.vec_total_sizes.append(scales[i]["size"])

  def create_cv(self, k):
    path, data_type, channels = self.paths[k]
    provenance = self.provenance 
    info = deepcopy(self.info)
    info['data_type'] = data_type
    info['num_channels'] = channels
    self.read[k] = CV(path, mkdir=False, info=info, provenance=provenance, **self.read_kwargs)
    self.write[k] = CV(path, mkdir=True, info=info, provenance=provenance, **self.write_kwargs)

  def add_path(self, k, path, data_type='uint8', num_channels=1):
    self.paths[k] = (path, data_type, num_channels)

  def create_paths(self):
    for k in self.paths.keys():
      self.create_cv(k)

  def get_composed_cv(self, compose_start, inverse, for_read):
    k = self.get_composed_key(compose_start, inverse)
    if for_read:
      return self.for_read(k)
    else:
      return self.for_write(k)

  def get_composed_key(self, compose_start, inverse):
    k = 'vvote_F{0}'.format(self.suffix)
    if inverse:
      k = 'vvote_invF{0}'.format(self.suffix)
    return '{0}_{1:04d}'.format(k, compose_start)
  
  def add_composed_cv(self, compose_start, inverse):
    """Create CloudVolume for storing composed vector fields

    Args
       compose_start: int, indicating the earliest section used for composing
       inverse: bool indicating whether composition aligns COMPOSE_START to Z (True),
        or Z to COMPOSE_START (False)
    """
    k = self.get_composed_key(compose_start, inverse)
    path = join(self.root, 'composed', self.get_composed_key(compose_start, inverse))
    self.add_path(k, path, data_type='float32', num_channels=2)
    self.create_cv(k)

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
  def __init__(self, archive, max_displacement, crop,
               mip_range, high_mip_chunk, src_path, tgt_path, dst_path, 
               src_mask_path='', src_mask_mip=0, src_mask_val=1, 
               tgt_mask_path='', tgt_mask_mip=0, tgt_mask_val=1,
               align_across_z=1, disable_cuda=False, max_mip=12,
               render_low_mip=2, render_high_mip=9, is_Xmas=False, threads=5,
               max_chunk=(1024, 1024), max_render_chunk=(2048*2, 2048*2),
               skip=0, topskip=0, size=7, should_contrast=True, 
               disable_flip_average=False, write_intermediaries=False,
               upsample_residuals=False, old_upsample=False, old_vectors=False,
               ignore_field_init=False, z=0, tgt_radius=1, forward_matches_only=False,
               queue_name=None, p_render=False, dir_suffix='', **kwargs):
    if queue_name != None:
        self.task_handler = TaskHandler(queue_name)
        self.distributed  = True
    else:
        self.task_handler = None
        self.distributed  = False
    self.p_render = p_render
    self.process_high_mip = mip_range[1]
    self.process_low_mip  = mip_range[0]
    self.render_low_mip   = render_low_mip
    self.render_high_mip  = render_high_mip
    # self.high_mip         = max(self.render_high_mip, self.process_high_mip)
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
    info = DstDir.create_info(src_cv, mip_range, max_displacement)
    self.dst = {}
    self.tgt_radius = tgt_radius
    self.tgt_range = range(-tgt_radius, tgt_radius+1)
    if forward_matches_only:
      self.tgt_range = range(tgt_radius+1)
    for i in self.tgt_range:
      if i > 0:
        path = '{0}/z_{1}'.format(dst_path, abs(i))
      elif i < 0:
        path = '{0}/z_{1}i'.format(dst_path, abs(i))
      else: 
        path = dst_path
      self.dst[i] = DstDir(path, info, provenance, suffix=dir_suffix)

    self.net = Process(archive, mip_range[0], is_Xmas=is_Xmas, cuda=True, 
                       dim=high_mip_chunk[0]+crop*2, skip=skip, 
                       topskip=topskip, size=size, 
                       flip_average=not disable_flip_average, old_upsample=old_upsample)

    self.normalizer = archive.preprocessor
    self.upsample_residuals = upsample_residuals
    self.pool = ThreadPool(threads)
    self.threads = threads

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
    
  def get_composed_field(self, src_z, tgt_z, F_cv, bbox, mip, 
                                inverse=False, relative=False, to_tensor=True):
    """Get composed field for Z_LIST using CloudVolume dirs at Z_OFFSET_LIST. Use field
    in BBOX at MIP. Use INVERSE to left-compose the next field in the list. Use RELATIVE
    to return a vector field in range [-1,1], and use TO_TENSOR to return a Tensor object.
    """
    #z_offset = src_z - tgt_z
    z_offset =  tgt_z - src_z
    f_cv = self.dst[z_offset].for_read('field')
    if inverse:
      f_z, F_z = src_z, src_z 
    else:
      f_z, F_z = src_z, tgt_z
    f = self.get_field(f_cv, f_z, bbox, mip, relative=True, to_tensor=to_tensor)
    F = self.get_field(F_cv, F_z, bbox, mip, relative=True, to_tensor=to_tensor)
    if inverse:
      F = self.compose_fields(f, F)
    else:
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

  def save_residual_patch(self, cv, z, res, bbox, mip):
    print ("Saving residual patch {} at MIP {}".format(bbox.__str__(mip=0), mip))
    v = res * (res.shape[-2] / 2) * (2**mip)
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
    #print("-----------------mip", mip , "high_mip", self.process_high_mip)
    #if (self.process_high_mip > mip):
    #    high_mip_scale = 2**(self.process_high_mip - mip)
    #else:
    #    high_mip_scale = 1
    high_mip_scale = 1
    processing_chunk = (int(self.high_mip_chunk[0] * high_mip_scale),
                        int(self.high_mip_chunk[1] * high_mip_scale))
    #print("--------processing_chunk", processing_chunk)
    if not render and (processing_chunk[0] > self.max_chunk[0]
                      or processing_chunk[1] > self.max_chunk[1]):
      processing_chunk = self.max_chunk
    elif render and (processing_chunk[0] > self.max_render_chunk[0]
                     or processing_chunk[1] > self.max_render_chunk[1]):
      processing_chunk = self.max_render_chunk
    for xs in range(calign_x_range[0], calign_x_range[1], processing_chunk[0]):
      for ys in range(calign_y_range[0], calign_y_range[1], processing_chunk[1]):
        chunks.append(BoundingBox(xs, xs + processing_chunk[0],
                                 ys, ys + processing_chunk[1],
                                 mip=mip, max_mip=self.max_mip)) #self.high_mip))

    return chunks

  def vector_vote(self, z, read_F_cv, write_F_cv, bbox, mip, inverse, T=1):
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
    for z_offset in range(1, self.tgt_radius+1):
      src_z = z
      tgt_z = src_z - z_offset
      if inverse:
        src_z, tgt_z = tgt_z, src_z 
      F = self.get_composed_field(src_z, tgt_z, read_F_cv, bbox, mip, 
                                  inverse=inverse, relative=False, to_tensor=True)
      fields.append(F)

    field = vector_vote(fields, T=T)
    self.save_vector_patch(write_F_cv, z, field, bbox, mip)

    # if self.write_intermediaries:
    #   self.save_image_patch('diffs', diffs.cpu().numpy(), bbox, mip, to_uint8=False)
    #   self.save_image_patch('diff_weights', diffs.cpu().numpy(), bbox, mip, to_uint8=False)
    #   self.save_image_patch('weights', diffs.cpu().numpy(), bbox, mip, to_uint8=False)

  def invert_field(self, z, src_cv, dst_cv, out_bbox, mip):
    """Compute the inverse vector field for a given OUT_BBOX
    """
    crop = self.crop_amount
    precrop_bbox = deepcopy(out_bbox)
    precrop_bbox.uncrop(crop, mip=mip)
    f = self.get_field(src_cv, z, precrop_bbox, mip, 
                           relative=True, to_tensor=True)
    print('invert_field shape: {0}'.format(f.shape))
    start = time()
    invf = invert(f)[:,crop:-crop, crop:-crop,:]    
    end = time()
    print (": {} sec".format(end - start))
    # assert(torch.all(torch.isnan(invf)))
    self.save_residual_patch(dst_cv, z, invf, out_bbox, mip) 

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

    # if self.write_intermediaries and residuals is not None and cum_residuals is not None:
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
    #print("++++++++++++++++abs_to_rel x_size and y_size", patch.x_size(mip=0), patch.y_size(mip=0))
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

  def warp_patch_at_low_mip(self, src_z, field_cv, field_z, bbox, image_mip, vector_mip):
    """Non-chunk warping

    From BBOX at MIP, warp image at SRC_Z in CloudVolume SRC_CV using
    field at FIELD_Z in CloudVolume FIELD_CV.
    """
    influence_bbox = deepcopy(bbox)
    influence_bbox.uncrop(self.max_displacement, mip=0)
    start = time()
    #print("image_mip is", image_mip, "vector_mip is", vector_mip) 
    field = self.get_field(field_cv, field_z, influence_bbox, vector_mip, 
                           relative=True, to_tensor=True)
     
    #print("field shape",field.shape)
    field_new = upsample(vector_mip - image_mip)(field.permute(0,3,1,2))
    mip_field = field_new.permute(0,2,3,1)
    mip_disp = int(self.max_displacement / 2**image_mip)
    #print("mip_field shape", mip_field.shape)
    #print("image_mip",image_mip, "vector_mip", vector_mip, "mip_dis is ", mip_disp)
    #print("bbox is ", bbox.__str__(mip=0), "influence_bbox is", influence_bbox.__str__(mip=0))
    src_cv = self.src['src_img']
    image = self.get_image(src_cv, src_z, influence_bbox, image_mip, 
                           adjust_contrast=False, to_tensor=True)
    #print("image shape", image.shape)
    if 'src_mask' in self.src:
      mask_cv = self.src['src_mask']
      mask = self.get_mask(mask_cv, src_z, influence_bbox, 
                           src_mip=self.src.src_mask_mip,
                           dst_mip=image_mip, valid_val=self.src.src_mask_val)
      image = image.masked_fill_(mask, 0)

    # print('warp_patch shape {0}'.format(image.shape))
    # no need to warp if flow is identity since warp introduces noise
    if torch.min(mip_field) != 0 or torch.max(mip_field) != 0:
      image = gridsample_residual(image, mip_field, padding_mode='zeros')
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
    #print("----------------z is", z, "save image patch at mip", mip, "range", x_range, y_range, "range at mip0", bbox.x_range(mip=0), bbox.y_range(mip=0))
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
      if self.normalizer is not None:
        data = self.normalizer(data).reshape(data.shape)
    # convert to tensor if requested, or if up/downsampling required
    if to_tensor | (src_mip != dst_mip):
      if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
      data = data.to(device=self.device)
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
    if self.distributed and len(chunks) > self.threads * 4:
        for i in range(0, len(chunks), self.threads * 4):
            task_patches = []
            for j in range(i, min(len(chunks), i + self.threads * 4)):
                task_patches.append(chunks[j])
            copy_task = make_copy_task_message(z, dst_cv, dst_z, task_patches, mip=mip)
            self.task_handler.send_message(copy_task)
        self.task_handler.wait_until_ready()
    else: 
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
    print('Rendering src_z={0} @ MIP{1} to dst_z={2}'.format(src_z, mip, dst_z), flush=True)
    start = time()
    chunks = self.break_into_chunks(bbox, self.dst[0].dst_chunk_sizes[mip],
                                    self.dst[0].dst_voxel_offsets[mip], mip=mip, render=True)
    if self.distributed:
        for i in range(0, len(chunks), self.threads):
            task_patches = []
            for j in range(i, min(len(chunks), i + self.threads)):
                task_patches.append(chunks[j])
            render_task = make_render_task_message(src_z, field_cv, field_z, task_patches, 
                                                   mip, dst_cv, dst_z)
            self.task_handler.send_message(render_task)
        self.task_handler.wait_until_ready()
    else:
        def chunkwise(patch_bbox):
          warped_patch = self.warp_patch(src_z, field_cv, field_z, patch_bbox, mip)
          # print('warp_image render.shape: {0}'.format(warped_patch.shape))
          self.save_image_patch(dst_cv, dst_z, warped_patch, patch_bbox, mip)
        self.pool.map(chunkwise, chunks)
    end = time()
    print (": {} sec".format(end - start))

  def low_mip_render(self, src_z, field_cv, field_z, dst_cv, dst_z, bbox, image_mip, vector_mip):
    start = time()
    chunks = self.break_into_chunks(bbox, self.dst[0].dst_chunk_sizes[image_mip],
                                    self.dst[0].dst_voxel_offsets[image_mip], mip=image_mip, render=True)
    print("low_mip_render at MIP{0} ({1} chunks)".format(image_mip,len(chunks)))
    for c in chunks:
        print(">>>>>>>>>chunks in low_mip_render: ", c.__str__(mip=0))
    if self.distributed:
        for i in range(0, len(chunks), self.threads):
            task_patches = []
            for j in range(i, min(len(chunks), i + self.threads)):
                task_patches.append(chunks[j])
            render_task = make_render_low_mip_task_message(src_z, field_cv, field_z, 
                                                           task_patches, image_mip, 
                                                           vector_mip, dst_cv, dst_z)
            self.task_handler.send_message(render_task)
        self.task_handler.wait_until_ready()
    else:
        def chunkwise(patch_bbox):
          warped_patch = self.warp_patch_at_low_mip(src_z, field_cv, field_z, patch_bbox, image_mip, vector_mip)
          print('warp_image render.shape: {0}'.format(warped_patch.shape))
          self.save_image_patch(dst_cv, dst_z, warped_patch, patch_bbox, image_mip)
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
      if self.distributed and len(chunks) > self.threads * 4:
          for i in range(0, len(chunks), self.threads * 4):
              task_patches = []
              for j in range(i, min(len(chunks), i + self.threads * 4)):
                  task_patches.append(chunks[j])
              downsample_task = make_downsample_task_message(cv, z, task_patches, mip=m)
              self.task_handler.send_message(downsample_task)
          self.task_handler.wait_until_ready()
      else:
          def chunkwise(patch_bbox):
            print ("Downsampling {} to mip {}".format(patch_bbox.__str__(mip=0), m))
            downsampled_patch = self.downsample_patch(cv, z, patch_bbox, m-1)
            self.save_image_patch(cv, z, downsampled_patch, patch_bbox, m)
          self.pool.map(chunkwise, chunks)

  def render_section_all_mips(self, src_z, field_cv, field_z, dst_cv, dst_z, bbox, mip):
    self.render(src_z, field_cv, field_z, dst_cv, dst_z, bbox, self.render_low_mip)
    self.downsample(dst_cv, dst_z, bbox, self.render_low_mip, self.render_high_mip)
  
  def render_to_low_mip(self, src_z, field_cv, field_z, dst_cv, dst_z, bbox, image_mip, vector_mip):
      self.low_mip_render(src_z, field_cv, field_z, dst_cv, dst_z, bbox, image_mip, vector_mip)
      self.downsample(dst_cv, dst_z, bbox, image_mip, self.render_high_mip)

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
      print ("compute residuals between {} to slice {} at mip {} ({} chunks)".
             format(src_z, tgt_z, m, len(chunks)), flush=True)
      if self.distributed:
        for patch_bbox in chunks:
          residual_task = make_residual_task_message(src_z, tgt_z, patch_bbox, mip=m)
          self.task_handler.send_message(residual_task)
        if not self.p_render:
          self.task_handler.wait_until_ready()
      else:
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

  def invert_field_chunkwise(self, z, src_cv, dst_cv, bbox, mip):
    """Chunked-processing of vector field inversion 
    
    Args:
       z: section of fields to weight
       src_cv: CloudVolume for forward field
       dst_cv: CloudVolume for inverted field
       bbox: boundingbox of region to process
       mip: field MIP level
    """
    start = time()
    chunks = self.break_into_chunks(bbox, self.dst[0].vec_chunk_sizes[mip],
                                    self.dst[0].vec_voxel_offsets[mip], mip=mip)
    print("Vector field inversion for slice {0} @ MIP{1} ({2} chunks)".
           format(z, mip, len(chunks)), flush=True)
    if self.distributed:
        for patch_bbox in chunks:
          invert_task = make_invert_field_task_message(z, src_cv, dst_cv, patch_bbox, mip)
          self.task_handler.send_message(invert_task)
    else: 
    #for patch_bbox in chunks:
        def chunkwise(patch_bbox):
          self.invert_field(z, src_cv, dst_cv, patch_bbox, mip)
        self.pool.map(chunkwise, chunks)
    end = time()
    print (": {} sec".format(end - start))

  def vector_vote_chunkwise(self, z, read_F_cv, write_F_cv, bbox, mip, inverse, T=-1):
    """Chunked-processing of vector voting
    
    Args:
       z: section of fields to weight
       read_F_cv: CloudVolume with the vectors to compose against
       write_F_cv: CloudVolume where the resulting vectors will be written 
       bbox: boundingbox of region to process
       mip: field MIP level
       T: softmin temperature (default will be 2**mip)
    """
    start = time()
    chunks = self.break_into_chunks(bbox, self.dst[0].vec_chunk_sizes[mip],
                                    self.dst[0].vec_voxel_offsets[mip], mip=mip)
    print("Vector voting for slice {0} @ MIP{1} {2} ({3} chunks)".
           format(z, mip, 'INVERSE' if inverse else 'FORWARD', len(chunks)), flush=True)
    
    if self.distributed:
        for patch_bbox in chunks:
            vector_vote_task = make_vector_vote_task_message(z, read_F_cv, write_F_cv,
                                                             patch_bbox, mip, inverse, T) 
            self.task_handler.send_message(vector_vote_task)
        self.task_handler.wait_until_ready()
    #for patch_bbox in chunks:
    else:
        def chunkwise(patch_bbox):
            self.vector_vote(z, read_F_cv, write_F_cv, patch_bbox, mip, inverse=inverse, T=T)
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
        tgt_z = src_z - z_offset
        self.compute_section_pair_residuals(src_z, tgt_z, bbox)
        if render:
          field_cv = self.dst[z_offset].for_read('field')
          dst_cv = self.dst[z_offset].for_write('dst_img')
          self.render_section_all_mips(src_z, field_cv, src_z, dst_cv, tgt_z, bbox, mip)

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
    if self.p_render:
        self.task_handler.wait_until_ready()
 
  def compose_pairwise(self, z_range, compose_start, bbox, mip, 
                             forward_compose=True, inverse_compose=True):
    """Combine pairwise vector fields in TGT_RADIUS using vector voting, while composing
    with earliest section at COMPOSE_START.

    Args
       z_range: list of ints (assumed to be monotonic & sequential)
       compose_start: int of earliest section used in composition
       bbox: BoundingBox defining chunk region
       mip: int for MIP level of data
       forward_compose: bool, indicating whether to compose with forward transforms
       inverse_compose: bool, indicating whether to compose with inverse transforms
    """
    self.total_bbox = bbox
    T = 2**mip
    print('softmin temp: {0}'.format(T))
    if forward_compose:
      self.dst[0].add_composed_cv(compose_start, inverse=False)
    if inverse_compose: 
      self.dst[0].add_composed_cv(compose_start, inverse=True)
    for z in z_range:
      write_F_k = self.dst[0].get_composed_key(compose_start, inverse=False)
      write_invF_k = self.dst[0].get_composed_key(compose_start, inverse=True)
      read_F_k = write_F_k
      read_invF_k = write_invF_k
       
      if forward_compose:
        read_F_cv = self.dst[0].for_read(read_F_k)
        write_F_cv = self.dst[0].for_write(write_F_k)
        self.vector_vote_chunkwise(z, read_F_cv, write_F_cv, bbox, mip, inverse=False, T=T)
      if inverse_compose:
        read_F_cv = self.dst[0].for_read(read_invF_k)
        write_F_cv = self.dst[0].for_write(write_invF_k)
        self.vector_vote_chunkwise(z, read_F_cv, write_F_cv, bbox, mip, inverse=True, T=T)

  def get_neighborhood(self, z, F_cv, bbox, mip):
    """Compile all vector fields that warp neighborhood in TGT_RANGE to Z

    Args
       z: int for index of SRC section
       F_cv: CloudVolume with fields 
       bbox: BoundingBox defining chunk region
       mip: int for MIP level of data
    """
    fields = []
    z_range = [z+z_offset for z_offset in range(self.tgt_radius + 1)]
    for k, tgt_z in enumerate(z_range):
      F = self.get_field(F_cv, tgt_z, bbox, mip, relative=True, to_tensor=True)
      fields.append(F)
    return torch.cat(fields, 0)
 
  def shift_neighborhood(self, Fs, z, F_cv, bbox, mip, keep_first=False): 
    """Shift field neighborhood by dropping earliest z & appending next z
  
    Args
       invFs: 4D torch tensor of inverse composed vector vote fields
       z: int representing the z of the input invFs. invFs will be shifted to z+1.
       F_cv: CloudVolume where next field will be loaded 
       bbox: BoundingBox representing xy extent of invFs
       mip: int for data resolution of the field
    """
    next_z = z + self.tgt_radius + 1
    next_F = self.get_field(F_cv, next_z, bbox, mip, relative=True, to_tensor=True)
    if keep_first:
      return torch.cat((Fs, next_F), 0)
    else:
      return torch.cat((Fs[1:, ...], next_F), 0)

  def regularize_z(self, z_range, dir_z, bbox, mip, sigma=1.4):
    """For a given chunk, temporally regularize each Z in Z_RANGE
    
    Make Z_RANGE as large as possible to avoid IO: self.shift_field
    is called to add and remove the newest and oldest sections.

    Args
       z_range: list of ints (assumed to be a contiguous block)
       overlap: int for number of sections that overlap with a chunk
       bbox: BoundingBox defining chunk region
       mip: int for MIP level of data
       sigma: float standard deviation of the Gaussian kernel used for the
        weighted average inverse
    """
    block_size = len(z_range)
    overlap = self.tgt_radius
    curr_block = z_range[0]
    next_block = curr_block + block_size
    self.dst[0].add_composed_cv(curr_block, inverse=False)
    self.dst[0].add_composed_cv(curr_block, inverse=True)
    self.dst[0].add_composed_cv(next_block, inverse=False)
    F_cv = self.dst[0].get_composed_cv(curr_block, inverse=False, for_read=True)
    invF_cv = self.dst[0].get_composed_cv(curr_block, inverse=True, for_read=True)
    next_cv = self.dst[0].get_composed_cv(next_block, inverse=False, for_read=False)
    z = z_range[0]
    invFs = self.get_neighborhood(z, invF_cv, bbox, mip)
    bump_dims = np.asarray(invFs.shape)
    bump_dims[0] = len(self.tgt_range)
    full_bump = create_field_bump(bump_dims, sigma)
    bump_z = 3 

    for z in z_range:
      composed = []
      bump = full_bump[bump_z:, ...]
      print(z)
      print(bump.shape)
      print(invFs.shape)
      F = self.get_field(F_cv, z, bbox, mip, relative=True, to_tensor=True)
      avg_invF = torch.sum(torch.mul(bump, invFs), dim=0, keepdim=True)
      regF = self.compose_fields(avg_invF, F)
      self.save_residual_patch(next_cv, z, regF, bbox, mip)
      if z != z_range[-1]:
        invFs = self.shift_neighborhood(invFs, z, invF_cv, bbox, mip, 
                                        keep_first=bump_z > 0)
      bump_z = max(bump_z - 1, 0)

  def regularize_z_chunkwise(self, z_range, dir_z, bbox, mip, sigma=1.4):
    """Chunked-processing of temporal regularization 
    
    Args:
       z_range: int list, range of sections over which to regularize 
       dir_z: int indicating the z index of the CloudVolume dir
       bbox: BoundingBox of region to process
       mip: field MIP level
       sigma: float for std of the bump function 
    """
    start = time()
    # self.dst[0].add_composed_cv(compose_start, inverse=False)
    # self.dst[0].add_composed_cv(compose_start, inverse=True)
    chunks = self.break_into_chunks(bbox, self.dst[0].vec_chunk_sizes[mip],
                                    self.dst[0].vec_voxel_offsets[mip], mip=mip)
    print("Regularizing slice range {0} @ MIP{1} ({2} chunks)".
           format(z_range, mip, len(chunks)), flush=True)
    if self.distributed:
        for patch_bbox in chunks:
            regularize_task = make_regularize_task_message(z_range[0], z_range[-1],
                                                      dir_z, patch_bbox,
                                                      mip, sigma)
            self.task_handler.send_message(regularize_task)
        self.task_handler.wait_until_ready()
    else:
        #for patch_bbox in chunks:
        def chunkwise(patch_bbox):
          self.regularize_z(z_range, dir_z, patch_bbox, mip, sigma=sigma)
        self.pool.map(chunkwise, chunks)
    end = time()
    print (": {} sec".format(end - start))

  def handle_residual_task(self, message):
    source_z = message['source_z']
    target_z = message['target_z']
    patch_bbox = deserialize_bbox(message['patch_bbox'])
    mip = message['mip']
    self.compute_residual_patch(source_z, target_z, patch_bbox, mip)

  def handle_render_task(self, message):
    src_z = message['z']
    patches  = [deserialize_bbox(p) for p in message['patches']]
    field_cv = DCV(message['field_cv']) 
    mip = message['mip']
    field_z = message['field_z']
    dst_cv = DCV(message['dst_cv'])
    dst_z = message['dst_z']
    def chunkwise(patch_bbox):
      print ("Rendering {} at mip {}".format(patch_bbox.__str__(mip=0), mip),
              end='', flush=True)
      warped_patch = self.warp_patch(src_z, field_cv, field_z, patch_bbox, mip)
      self.save_image_patch(dst_cv, dst_z, warped_patch, patch_bbox, mip)
    self.pool.map(chunkwise, patches)

  def handle_render_task_low_mip(self, message):
    src_z = message['z']
    patches  = [deserialize_bbox(p) for p in message['patches']]
    field_cv = DCV(message['field_cv']) 
    image_mip = message['image_mip']
    vector_mip = message['vector_mip']
    field_z = message['field_z']
    dst_cv = DCV(message['dst_cv'])
    dst_z = message['dst_z']
    def chunkwise(patch_bbox):
      print ("Rendering {} at mip {}".format(patch_bbox.__str__(mip=0), image_mip),
              end='', flush=True)
      warped_patch = self.warp_patch_at_low_mip(src_z, field_cv, field_z, 
                                                patch_bbox, image_mip, vector_mip)
      self.save_image_patch(dst_cv, dst_z, warped_patch, patch_bbox, image_mip)
    self.pool.map(chunkwise, patches)

  def handle_prepare_task(self, message):
    z = message['z']
    patches  = [deserialize_bbox(p) for p in message['patches']]
    mip = message['mip']
    start_z = message['start_z']
    def chunkwise(patch_bbox):
      print ("Preparing source {} at mip {}".format(patch_bbox.__str__(mip=0), mip),
              end='', flush=True)
      warped_patch = self.warp_patch(self.src_ng_path, z, patch_bbox,
                                      (mip, self.process_high_mip), mip, start_z)
      self.save_image_patch(self.tmp_ng_path, warped_patch, z, patch_bbox, mip)

    self.pool.map(chunkwise, patches)

  def handle_compose_task(self, message):
    z = message['z']
    patches  = [deserialize_bbox(p) for p in message['patches']]
    mip = message['mip']
    start_z = message['start_z']
    def chunkwise(patch_bbox):
      print ("composing {} at mip {}".format(patch_bbox.__str__(mip=0), mip),
              end='', flush=True) 
      self.compose_field_task(z ,patch_bbox, (mip, self.process_high_mip), mip, start_z)
    self.pool.map(chunkwise, patches)


  def handle_copy_task(self, message):
    z = message['z']
    patches  = [deserialize_bbox(p) for p in message['patches']]
    mip = message['mip']
    dst_cv = DCV(message['dst_cv'])
    dst_z = message['dst_z']
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
    self.pool.map(chunkwise, patches)

  def handle_downsample_task(self, message):
    z = message['z']
    cv = DCV(message['cv'])
    patches  = [deserialize_bbox(p) for p in message['patches']]
    mip = message['mip']
    def chunkwise(patch_bbox):
      downsampled_patch = self.downsample_patch(cv, z, patch_bbox, mip - 1)
      self.save_image_patch(cv, z, downsampled_patch, patch_bbox, mip)
    self.pool.map(chunkwise, patches)

  def handle_vector_vote(self, message):
      z = message['z']
      read_F_cv = DCV(message['read_F_cv'])
      write_F_cv =DCV(message['write_F_cv'])
      #chunks = [deserialize_bbox(p) for p in message['patch_bbox']]
      chunks = deserialize_bbox(message['patch_bbox'])
      mip = message['mip']
      inverse = message['inverse']
      T = message['T']
      self.vector_vote(z, read_F_cv, write_F_cv, chunks, mip, inverse=inverse, T=T)

  def handle_regularize(self, message):
      z_start = message['z_start']
      z_end = message['z_end']
      compose_start = message['compose_start']
      patch_bbox = deserialize_bbox(message['patch_bbox'])
      mip = message['mip']
      sigma = message['sigma']
      z_range = range(z_start, z_end)
      self.regularize_z(z_range, compose_start, patch_bbox, mip, sigma=sigma)

  def handle_invert(self, message):
      src_cv = DCV(message['src_cv'])
      dst_cv = DCV(message['dst_cv'])
      patch_bbox = deserialize_bbox(message['patch_bbox'])
      mip = message['mip']
      self.invert_field(z, src_cv, dst_cv, patch_bbox, mip)

  def handle_task_message(self, message):
    #message types:
    # -compute residual
    # -prerender future target
    # -render final result
    # -downsample
    # -copy
    #import pdb; pdb.set_trace()
    body = json.loads(message['Body'])
    task_type = body['type']
    if task_type == 'residual_task':
      self.handle_residual_task(body)
    elif task_type == 'render_task':
      self.handle_render_task(body)
    elif task_type == 'render_task_low_mip':
      self.handle_render_task_low_mip(body)
    elif task_type == 'compose_task':
      self.handle_compose_task(body)
    elif task_type == 'copy_task':
      print("copy_task ----")
      self.handle_copy_task(body)
    elif task_type == 'downsample_task':
      self.handle_downsample_task(body)
    elif task_type == 'prepare_task':
      self.handle_prepare_task(body)
    elif task_type == 'vector_vote_task':
      self.handle_vector_vote(body)
    elif task_type == 'regularize_task':
      self.handle_regularize(body)     
    elif task_type == 'invert_task':
      self.handle_invert(body)
    else:
      raise Exception("Unsupported task type '{}' received from queue '{}'".format(task_type,
                                                                 self.task_handler.queue_name))

  def listen_for_tasks(self, stack_start, stack_size ,bbox, forward_compose, inverse_compose, compose_start):
    self.total_bbox = bbox
    self.zs = stack_start
    self.end_section = stack_start + stack_size
    self.num_section = stack_size
    if forward_compose:
      self.dst[0].add_composed_cv(compose_start, inverse=False)
    if inverse_compose: 
      self.dst[0].add_composed_cv(compose_start, inverse=True)
    while (True):
      message = self.task_handler.get_message()
      if message != None:
        print ("Got a job")
        s = time()
        #self.task_handler.purge_queue()
        self.handle_task_message(message)
        self.task_handler.delete_message(message)
        e = time()
        print ("Done: {} sec".format(e - s))
      else:
        sleep(3)
        print ("Waiting for jobs...") 


