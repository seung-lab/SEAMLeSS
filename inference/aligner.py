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
from helpers import save_chunk, crop, upsample, gridsample_residual, np_downsample

from skimage.morphology import disk as skdisk
from skimage.filters.rank import maximum as skmaximum

from boundingbox import BoundingBox

from pathos.multiprocessing import ProcessPool, ThreadPool
from threading import Lock

import torch.nn as nn

class AlignerDir():
  """Manager of output cloudvolumes required by the Aligner
  """
  def __init__(self, src_path, tgt_path, 
                     src_mask_path, tgt_mask_path, 
                     dst_path, mip_range):
    self.mip_range = mip_range
    self.paths = self.get_paths(dst_path)
    self.dst_chunk_sizes   = []
    self.dst_voxel_offsets = []
    self.vec_chunk_sizes   = []
    self.vec_voxel_offsets = []
    self.vec_total_sizes   = []
    self.vols = {}
    self.in_kwargs = {'bounded': False, 'fill_missing': True, 'progress': False}
    self.out_kwargs = {'bounded': False, 'fill_missing': True, 'progress': False, 
                  'autocrop': True, 'non_aligned_writes': False}
    self.vols['src_img'] = CV(src_path, **self.in_kwargs) 
    self.vols['src_mask'] = CV(src_mask_path, **self.in_kwargs) 
    self.vols['tgt_img'] = CV(tgt_path, **self.in_kwargs) 
    self.vols['tgt_mask'] = CV(tgt_mask_path, **self.in_kwargs) 
    self.src_mask_val = src_mask_val
    self.tgt_mask_val = tgt_mask_val
    self.src_mask_mip = src_mask_mip
    self.tgt_mask_mip = tgt_mask_mip
  
  def __getitem__(self, k):
    return self.vols[k]

  def create(self, max_offset, write_intermediaries):
    src_info = self.vols['src_img'][0].info
    m = len(src_info['scales'])
    each_factor = Vec(2,2,1)
    factor = Vec(2**m,2**m,1)
    for _ in self.mip_range: 
      src_cv.add_scale(factor)
      factor *= each_factor
      chunksize = src_info['scales'][-2]['chunk_sizes'][0] // each_factor
      src_info['scales'][-1]['chunk_sizes'] = [ list(map(int, chunksize)) ]

    # print(src_info)
    img_info = deepcopy(src_info)

    ##########################################################
    #### Create img info file
    ##########################################################
    chunk_size = img_info["scales"][0]["chunk_sizes"][0][0]
    dst_size_increase = max_offset
    if dst_size_increase % chunk_size != 0:
      dst_size_increase = dst_size_increase - (dst_size_increase % max_offset) + chunk_size
    scales = img_info["scales"]
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

    for k in ['dst_img', 'tmp_img']:
      self.cvs[k] = CV(self.paths[k], info=img_info, provenance=provenance, 
                                         mkdir=True, **self.out_kwargs)

    ##########################################################
    #### Create vec info file
    ##########################################################
    vec_info = deepcopy(img_info)
    vec_info["data_type"] = "float32"
    for i in range(len(vec_info["scales"])):
      vec_info["scales"][i]["chunk_sizes"][0][2] = 1
    vec_info['num_channels'] = 2

    # enc_dict = {x: 6*(x-self.process_low_mip)+12 for x in 
    #                range(self.process_low_mip, self.process_high_mip+1)} 

    scales = deepcopy(vec_info["scales"])

    for i in range(len(scales)):
      self.vec_chunk_sizes.append(scales[i]["chunk_sizes"][0][0:2])
      self.vec_voxel_offsets.append(scales[i]["voxel_offset"])
      self.vec_total_sizes.append(scales[i]["size"])

    self.cvs['field'] = CV(self.paths['field'], info=vec_info, 
                              provenance=provenance, mkdir=not self.ignore_field_init,
                              **self.out_kwargs)
    if write_intermediaries:
      for k in ['res', 'cumres', 'resup', 'cumresup']:
        self.cvs[k] = CV(self.paths[k], info=vec_info, provenance=provenance,
                              mkdir=True, **self.out_kwargs)

      # if i in enc_dict.keys():
      #   enc_info = deepcopy(vec_info)
      #   enc_info['num_channels'] = enc_dict[i]
      #   # enc_info['data_type'] = 'uint8'
      #   self.cvs['enc'] = CV(self.paths['enc'], info=enc_info, provenance=provenance, mkdir=True)

      wts_info = deepcopy(vec_info)
      wts_info['num_channels'] = 3
      # enc_info['data_type'] = 'uint8'
      for k in ['diffs', 'diff_weights', 'weights']:
        self.cvs[k] = CV(self.paths[k], info=vec_info, provenance=provenance, 
                         mkdir=True, **self.out_kwargs)

  def get_paths(self, dst_root):
    paths = {}
    paths['dst_root'] = root 
    paths['dst_img'] = join(root, 'image')
    paths['tmp_img'] = join(root, 'intermediate')
    paths['enc'] = join(root, 'enc')

    paths['res'] = join(root, 'vec')
    paths['cumres'] = join(root, 'cumulative_vec')
    paths['resup'] = join(root, 'vec_up')
    paths['cumresup'] = join(root, 'cumulative_vec_up')
    paths['field']   = join(root, 'field')
    paths['diffs']   = join(root, 'diffs')
    paths['diff_weights']   = join(root, 'diff_weights')
    paths['weights']   = join(root, 'weights')
    return paths

class Aligner:
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

    self.max_displacement = max_displacement
    self.crop_amount      = crop
    self.disable_cuda = disable_cuda
    self.device = torch.device('cpu') if disable_cuda else torch.device('cuda')

    self.offset_vols = {}
    self.tgt_range = range(1, tgt_radius)
    for i in range(tgt_radius): 
      path = '{0}/z_{1}'.format(dst_path, abs(i))
      if i == 0:
        path = dst_path
      self.offset_vols[i] = AlignerDir(src_path, tgt_path, 
                                       src_mask_path, tgt_mask_path, 
                                       dst_path, mip_range)
      self.offset_vols[i].create(mask_displacement, self.write_intermediaries)
    # set z_offset to 0 and self.vols to root
    self.z = 0
    self.set_z_offset(0)

    self.net = Process(model_path, mip_range[0], is_Xmas=is_Xmas, cuda=True, 
                       dim=high_mip_chunk[0]+crop*2, skip=skip, 
                       topskip=topskip, size=size, 
                       flip_average=not disable_flip_average, old_upsample=old_upsample)

    self.normalizer = Normalizer(min(5, mip_range[0])) 
    self.write_intermediaries = write_intermediaries
    self.upsample_residuals = upsample_residuals

    self.pool = ThreadPool(threads)

    self.img_cache = {}
    self.img_cache_lock = Lock()

  def set_z(self, z):
    self.z = z

  def set_z_offset(self, z_offset):
    self.z_offset = z_offset
    self.vols = self.offset_dsts[z_offset]

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

  def check_all_params(self):
    return True

  def get_upchunked_bbox(self, bbox, ng_chunk_size, offset, mip):
    raw_x_range = bbox.x_range(mip=mip)
    raw_y_range = bbox.y_range(mip=mip)

    x_chunk = ng_chunk_size[0]
    y_chunk = ng_chunk_size[1]

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

  def get_composed_field(self, bbox, mip, relative=False, to_tensor=True):
    orig_z_offset = self.z_offset
    field = get_field('field', bbox, mip, relative=True, to_tensor=to_tensor)
    self.set_z_offset(0)
    field_sf = get_field('field', bbox, mip, relative=True, to_tensor=to_tensor)
    self.set_z_offset(orig_z_offset)
    field_sf = self.blur_field(field_sf)
    field = field.permute(0,3,1,2)
    composed_field = field_sf + gridsample_residual(
        field, field_sf, padding_mode='border').permute(0,2,3,1)
    if not relative:
      composed_field = self.rel_to_abs_residual(composed_field, mip)
    return composed_field

  def blur_field(self, field, std=128):
    """Apply Gaussian with std to a vector field
    """
    regular_part_x = torch.from_numpy(scipy.ndimage.filters.gaussian_filter((field[...,0]), std)).unsqueeze(-1)
    regular_part_y = torch.from_numpy(scipy.ndimage.filters.gaussian_filter((field[...,1]), std)).unsqueeze(-1)
    #regular_part = self.gauss_filter(field.permute(3,0,1,2))
    #regular_part = torch.from_numpy(self.reg_field) 
    #field = decay_factor * field + (1 - decay_factor) * regular_part.permute(1,2,3,0) 
    #field = regular_part.permute(1,2,3,0) 
    field = torch.cat([regular_part_x,regular_part_y],-1)
    return field.to(device=self.device)

  def get_field(self, cv_key, bbox, mip, relative=False, to_tensor=True, use_root=False):
    z = self.z
    x_range = bbox.x_range(mip=mip)
    y_range = bbox.y_range(mip=mip)
    cv = self.vols[cv_key][mip]
    field = cv[x_range[0]:x_range[1], y_range[0]:y_range[1], z]
    res = np.expand_dims(np.squeeze(field), axis=0)
    if relative:
      res = self.abs_to_rel_residual(res, bbox, mip)
    if to_tensor:
      res = torch.from_numpy(res)
      return res.to(device=self.device)
    else:
      return res

  def save_vector_patch(self, cv_key, field, bbox, mip):
    z = self.z
    field = field.data.cpu().numpy() 
    x_range = bbox.x_range(mip=mip)
    y_range = bbox.y_range(mip=mip)
    field = np.squeeze(field)[:, :, np.newaxis, :]
    cv = self.vols[cv_key][mip]
    cv[x_range[0]:x_range[1], y_range[0]:y_range[1], z] = field

  def save_residual_patch(self, cv_key, res, crop, bbox, mip):
    print ("Saving residual patch {} at MIP {}".format(bbox.__str__(mip=0), mip))
    v = res * (res.shape[-2] / 2) * (2**mip)
    v = v[:,crop:-crop, crop:-crop,:]
    self.save_vector_patch(cv_key, v, bbox, mip)

  def break_into_chunks(self, bbox, ng_chunk_size, offset, mip, render=False):
    chunks = []
    raw_x_range = bbox.x_range(mip=mip)
    raw_y_range = bbox.y_range(mip=mip)

    x_chunk = ng_chunk_size[0]
    y_chunk = ng_chunk_size[1]

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

  def weight_fields(self, bbox, mip, T=1):
    """Calculate voting weights for the align across vector fields
    """
    fields = []
    for z_offset in self.tgt_range: 
      self.set_z_offset(z_offset)
      fields.append(self.get_composed_field(bbox, mip, relative=False))
    field = vector_vote(fields, T=T)
    self.set_z_offset(0)
    self.save_vector_patch('field', field, z, bbox, mip)

    if write_intermediaries:
      self.save_image_patch('diffs', diffs.cpu().numpy(), bbox, mip, to_uint8=False)
      self.save_image_patch('diff_weights', diffs.cpu().numpy(), bbox, mip, to_uint8=False)
      self.save_image_patch('weights', diffs.cpu().numpy(), bbox, mip, to_uint8=False)

  def compute_residual_patch(self, out_patch_bbox, mip, inverse=False):
    source_z = self.z
    target_z = self.z - self.z_offset
    if inverse:
      source_z, target_z = target_z, source_z
    print ("Computing residual for region {}.".format(out_patch_bbox.__str__(mip=0)), flush=True)
    precrop_patch_bbox = deepcopy(out_patch_bbox)
    precrop_patch_bbox.uncrop(self.crop_amount, mip=mip)

    if mip == self.process_high_mip:
      src_patch = self.get_image('src_img', precrop_patch_bbox, mip,
                                  adjust_contrast=True, to_tensor=True)
    else:
      src_patch = self.get_image('tmp_img', precrop_patch_bbox, mip,
                                  adjust_contrast=True, to_tensor=True)

    tgt_patch = self.get_image('tgt_img', precrop_patch_bbox, mip,
                                adjust_contrast=True, to_tensor=True) 

    if 'src_mask' in self.vols:
      src_mask = self.get_mask('src_mask', precrop_patch_bbox, 
                           src_mip=self.vols.src_mask_mip,
                           dst_mip=mip, valid_val=self.vols.src_mask_val)
      src_patch = src_patch.masked_fill_(src_mask, 0)
    if 'tgt_mask' in self.vols:
      tgt_mask = self.get_mask('tgt_mask', precrop_patch_bbox, 
                           src_mip=self.vols.tgt_mask_mip,
                           dst_mip=mip, valid_val=self.vols.tgt_mask_val)
      tgt_patch = tgt_patch.masked_fill_(tgt_mask, 0)

    X = self.net.process(src_patch, tgt_patch, mip, crop=self.crop_amount, 
                                                 old_vectors=self.old_vectors)
    field, residuals, encodings, cum_residuals = X

    # save the final vector field for warping
    self.save_vector_patch(field, source_z, out_patch_bbox, mip)

    if self.write_intermediaries:
      mip_range = range(self.process_low_mip+self.size-1, self.process_low_mip-1, -1)
      for res_mip, res, cumres in zip(mip_range, residuals[1:], cum_residuals[1:]):
          crop = self.crop_amount // 2**(res_mip - self.process_low_mip)   
          self.save_residual_patch('res', res, crop, 
                                   source_z, out_patch_bbox, res_mip)
          self.save_residual_patch('cumres', cumres, crop, 
                                   source_z, out_patch_bbox, res_mip)
          if self.upsample_residuals:
            crop = self.crop_amount   
            res = self.scale_residuals(res, res_mip, self.process_low_mip)
            self.save_residual_patch('resup', res, crop, source_z, out_patch_bbox, 
                                     self.process_low_mip)
            cumres = self.scale_residuals(cumres, res_mip, self.process_low_mip)
            self.save_residual_patch('cumresup', cumres, crop, 
                                     source_z, out_patch_bbox, 
                                     self.process_low_mip)

      # print('encoding size: {0}'.format(len(encodings)))
      for k, enc in enumerate(encodings):
          mip = self.process_low_mip + k
          # print('encoding shape @ idx={0}, mip={1}: {2}'.format(k, mip, enc.shape))
          crop = self.crop_amount // 2**k
          enc = enc[:,:,crop:-crop, crop:-crop].permute(2,3,0,1)
          enc = enc.data.cpu().numpy()
          
          def write_encodings(j_slice, z):
            x_range = out_patch_bbox.x_range(mip=mip)
            y_range = out_patch_bbox.y_range(mip=mip)
            patch = enc[:, :, :, j_slice]
            # uint_patch = (np.multiply(patch, 255)).astype(np.uint8)
            cv(self.paths['enc'][mip], 
                mip=mip, bounded=False, 
                fill_missing=True, autocrop=True, 
                progress=False, provenance={})[x_range[0]:x_range[1],
                                y_range[0]:y_range[1], z, j_slice] = patch 
  
          # src_image encodings
          write_encodings(slice(0, enc.shape[-1] // 2), source_z)
          # dst_image_encodings
          write_encodings(slice(enc.shape[-1] // 2, enc.shape[-1]), target_z)

  def rel_to_abs_residual(self, field, mip):    
      return field * (field.shape[-2] / 2) * (2**mip)

  def abs_to_rel_residual(self, abs_residual, patch, mip):
    x_fraction = patch.x_size(mip=0) * 0.5
    y_fraction = patch.y_size(mip=0) * 0.5

    rel_residual = deepcopy(abs_residual)
    rel_residual[0, :, :, 0] /= x_fraction
    rel_residual[0, :, :, 1] /= y_fraction
    return rel_residual

  def calc_image_mean_field(self, image, field, cid):
      for i in range(0,image.shape[0]):
          for j in range(0,image.shape[1]):
              if(image[i,j]!=0):
                  self.image_pixels_sum[cid] +=1
                  self.field_sf_sum[cid] += field[i,j]

  def get_bbox_id(self, in_bbox, mip):
    raw_x_range = self.total_bbox.x_range(mip=mip)
    raw_y_range = self.total_bbox.y_range(mip=mip)

    x_chunk = self.vols.dst_chunk_sizes[mip][0]
    y_chunk = self.vols.dst_chunk_sizes[mip][1]

    x_offset = self.vols.dst_voxel_offsets[mip][0]
    y_offset = self.vols.dst_voxel_offsets[mip][1]

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
  def warp_patch(self, bbox, mip, start_z=-1):
    z = self.z
    influence_bbox = deepcopy(bbox)
    influence_bbox.uncrop(self.max_displacement, mip=0)
    start = time()
    
    field = self.get_field(influence_bbox, mip, relative=True, to_tensor=True)
    # print('field0.shape: {0}'.format(field.shape))
    field = self.scale_residuals(field, mip+1, mip)
    # print('field1.shape: {0}'.format(field.shape))
    mip_disp = int(self.max_displacement / 2**mip)
    # print('mip_disp: {0}'.format(mip_disp))
    image = self.get_image('src_img', influence_bbox, mip, 
                           adjust_contrast=False, to_tensor=True)
    # print('warp_image image0.shape: {0}'.format(image.shape))
    if 'src_mask' in self.vols:
      mask = self.get_mask('src_mask', influence_bbox, 
                           src_mip=self.vols.src_mask_mip,
                           dst_mip=mip, valid_val=self.vols.src_mask_val)
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

  def downsample_patch(self, cv_key, bbox, mip):
    data = self.get_image(cv_key, bbox, mip - 1, 
                              adjust_contrast=False, to_tensor=True)
    data = interpolate(data, scale_factor=0.5, mode='bilinear')
    return data.cpu().numpy()

  ## Data saving
  def save_image_patch(self, cv_key, float_patch, bbox, mip, to_uint8=True):
    z = self.z - self.z_offset
    x_range = bbox.x_range(mip=mip)
    y_range = bbox.y_range(mip=mip)
    patch = np.transpose(float_patch, (3,2,1,0))
    if to_uint8:
      patch = (np.multiply(patch, 255)).astype(np.uint8)
    cv = self.vols[cv_key][mip]
    cv[x_range[0]:x_range[1], y_range[0]:y_range[1], z] = patch

  def scale_residuals(self, res, src_mip, dst_mip):
    print('Upsampling residuals from MIP {0} to {1}'.format(src_mip, dst_mip))
    up = nn.Upsample(scale_factor=2, mode='bilinear')
    for m in range(src_mip, dst_mip, -1):
      res = up(res.permute(0,3,1,2)).permute(0,2,3,1)
    return res

  ## Data loading
  def dilate_mask(self, mask, radius=5):
    return skmaximum(np.squeeze(mask).astype(np.uint8), skdisk(radius)).reshape(mask.shape).astype(np.bool)
    
  def missing_data_mask(self, img, bbox, mip):
    (img_xs, img_xe), (img_ys, img_ye) = bbox.x_range(mip=mip), bbox.y_range(mip=mip)
    (total_xs, total_xe), (total_ys, total_ye) = self.total_bbox.x_range(mip=mip), self.total_bbox.y_range(mip=mip)
    xs_inset = max(0, total_xs - img_xs)
    xe_inset = max(0, img_xe - total_xe)
    ys_inset = max(0, total_ys - img_ys)
    ye_inset = max(0, img_ye - total_ye)
    mask = np.logical_or(img == 0, img >= 253)
    
    fov_mask = np.ones(mask.shape).astype(np.bool)
    if xs_inset > 0:
      fov_mask[:xs_inset] = False
    if xe_inset > 0:
      fov_mask[-xe_inset:] = False
    if ys_inset > 0:
      fov_mask[:,:ys_inset] = False
    if ye_inset > 0:
      fov_mask[:,-ye_inset:] = False

    return np.logical_and(fov_mask, mask)
    
#   def supplement_target_with_backup(self, target, still_missing_mask, backup, bbox, mip):
#     backup_missing_mask = self.missing_data_mask(backup, bbox, mip)
#     fill_in = backup_missing_mask < still_missing_mask
#     target[fill_in] = backup[fill_in]
# 
#   def check_image_cache(self, path, bbox, mip):
#     with self.img_cache_lock:
#       output = -1 * np.ones((1,1,bbox.x_size(mip), bbox.y_size(mip)))
#       for key in self.img_cache:
#         other_path, other_bbox, other_mip = key[0], key[1], key[2]
#         if other_mip == mip and other_path == path:
#           if bbox.intersects(other_bbox):
#             xs, ys, xsz, ysz = other_bbox.insets(bbox, mip)
#             output[:,:,xs:xs+xsz,ys:ys+ysz] = self.img_cache[key]
#     if np.min(output > -1):
#       print('hit')
#       return output
#     else:
#       return None
# 
#   def add_to_image_cache(self, path, bbox, mip, data):
#     with self.img_cache_lock:
#       self.img_cache[(path, bbox, mip)] = data

  def get_mask(self, cv_key, bbox, src_mip, dst_mip, valid_val, to_tensor=True):
    data = self.get_data(cv_key, bbox, src_mip=src_mip, dst_mip=dst_mip, 
                             to_float=False, adjust_contrast=False, 
                             to_tensor=to_tensor)
    return data == valid_val

  def get_image(self, cv_key, bbox, mip, adjust_contrast=False, to_tensor=True):
    return self.get_data(cv_key, bbox, src_mip=mip, dst_mip=mip, to_float=True, 
                             adjust_contrast=adjust_contrast, to_tensor=to_tensor)

  def get_data(self, cv_key, bbox, src_mip, dst_mip, 
                     to_float=True, adjust_contrast=False, to_tensor=True):
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
    z = self.z
    cv = self.vols[cv_key][src_mip]
    x_range = bbox.x_range(mip=src_mip)
    y_range = bbox.y_range(mip=src_mip)
    data = cv[x_range[0]:x_range[1], y_range[0]:y_range[1], z] 
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
  def copy_section(self, bbox, mip):
    z = self.z
    print ("moving section {} mip {} to dest".format(z, mip), end='', flush=True)
    start = time()
    chunks = self.break_into_chunks(bbox, self.vols.dst_chunk_sizes[mip],
                                    self.vols.dst_voxel_offsets[mip], mip=mip, render=True)
    #for patch_bbox in chunks:
    def chunkwise(patch_bbox):

      if self.paths['src_mask']:
        raw_patch = self.get_image('src_img', patch_bbox, mip,
                                    adjust_contrast=False, to_tensor=True)
        raw_mask = self.get_mask('src_mask', precrop_patch_bbox, 
                                 src_mip=self.vols.src_mask_mip,
                                 dst_mip=mip, valid_val=self.vols.src_mask_val)
        raw_patch = raw_patch.masked_fill_(raw_mask, 0)
        raw_patch = raw_patch.cpu().numpy()
      else: 
        raw_patch = self.get_image('src_img', patch_bbox, mip,
                                    adjust_contrast=False, to_tensor=False)
      self.save_image_patch('dst_img', raw_patch, patch_bbox, mip)

    self.pool.map(chunkwise, chunks)

    end = time()
    print (": {} sec".format(end - start))

  def prepare_source(self, bbox, mip):
    print ("Prerendering mip {}".format(mip),
           end='', flush=True)
    start = time()

    chunks = self.break_into_chunks(bbox, self.vols.dst_chunk_sizes[mip],
                                    self.vols.dst_voxel_offsets[mip], mip=mip, render=True)

    def chunkwise(patch_bbox):
      warped_patch = self.warp_patch(patch_bbox, mip)
      self.save_image_patch('tmp_img', warped_patch, patch_bbox, mip)
    self.pool.map(chunkwise, chunks)
    end = time()
    print (": {} sec".format(end - start))

  def render(self, bbox, mip):
    z = self.z
    z_offset = self.z_offset
    print('Rendering z={0} with z_offset={1} @ MIP{2}'.format(z, z_offset, mip), flush=True)
    start = time()
    chunks = self.break_into_chunks(bbox, self.vols.dst_chunk_sizes[mip],
                                    self.vols.dst_voxel_offsets[mip], mip=mip, render=True)
    #print("\n total chunsk is ", len(chunks))
    #if (self.run_pairs and (z!=start_z)):
    #    total_chunks = len(chunks) 
    #    self.reg_field= np.sum(self.field_sf_sum, axis=0) / np.sum(self.image_pixels_sum)
    #    self.image_pixels_sum = np.zeros(total_chunks)
    #    self.field_sf_sum = np.zeros((total_chunks, 2), dtype=np.float32)

    def chunkwise(patch_bbox):
      warped_patch = self.warp_patch(patch_bbox, mip)
      print('warp_image render.shape: {0}'.format(warped_patch.shape))
      self.save_image_patch('dst_img', warped_patch, patch_bbox, mip)
    self.pool.map(chunkwise, chunks)
    end = time()
    print (": {} sec".format(end - start))

  def render_section_all_mips(self, bbox):
    self.render(bbox, self.render_low_mip)
    self.downsample(bbox, self.render_low_mip, self.render_high_mip)

  def downsample(self, bbox, source_mip, target_mip):
    print ("Downsampling {} from mip {} to mip {}".format(bbox.__str__(mip=0), source_mip, target_mip))
    for m in range(source_mip+1, target_mip + 1):
      chunks = self.break_into_chunks(bbox, self.vols.dst_chunk_sizes[m],
                                      self.vols.dst_voxel_offsets[m], mip=m, render=True)

      def chunkwise(patch_bbox):
        print ("Downsampling {} to mip {}".format(patch_bbox.__str__(mip=0), m))
        downsampled_patch = self.downsample_patch('dst_img', patch_bbox, m)
        self.save_image_patch('dst_img', downsampled_patch, patch_bbox, m)
      self.pool.map(chunkwise, chunks)

  def compute_section_pair_residuals(self, bbox, inverse=False):
    source_z = self.z
    target_z = self.z - self.offset_z
    if inverse:
      source_z, target_z = target_z, source_z
    for m in range(self.process_high_mip,  self.process_low_mip - 1, -1):
      start = time()
      chunks = self.break_into_chunks(bbox, self.vols.vec_chunk_sizes[m],
                                      self.vols.vec_voxel_offsets[m], mip=m)
      print ("Aligning slice {} to slice {} at mip {} ({} chunks)".
             format(source_z, target_z, m, len(chunks)), flush=True)

      #for patch_bbox in chunks:
      def chunkwise(patch_bbox):
      #FIXME Torch runs out of memory
      #FIXME batchify download and upload
        self.compute_residual_patch(patch_bbox, mip=m, inverse=inverse)
      self.pool.map(chunkwise, chunks)
      end = time()
      print (": {} sec".format(end - start))

      if m > self.process_low_mip:
          self.prepare_source(source_z, bbox, m - 1)
    
  def count_box(self, bbox, mip):    
    chunks = self.break_into_chunks(bbox, self.vols.dst_chunk_sizes[mip],
                                      self.vols.dst_voxel_offsets[mip], mip=mip, render=True)
    total_chunks = len(chunks)
    self.image_pixels_sum =np.zeros(total_chunks)
    self.field_sf_sum =np.zeros((total_chunks, 2), dtype=np.float32)


  def compute_weighted_field(self, bbox, mip, T=-1):
    """Chunked-processing of field weighting
    
    Args:
       z: section of fields to weight
       bbox: boundingbox of region to process
       mip: field MIP level
       T: softmin temperature (default will be 2**mip)
    """
    z = self.z
    start = time()
    chunks = self.break_into_chunks(bbox, self.vols.vec_chunk_sizes[mip],
                                    self.vols.vec_voxel_offsets[mip], mip=mip)
    print("Computing weighted field for slice {0} @ MIP{1} ({2} chunks)".
           format(z, mip, len(chunks)), flush=True)
    print('Writing vectors to {0}'.format(self.paths['field'][mip]))

    #for patch_bbox in chunks:
    def chunkwise(patch_bbox):
      self.weight_fields(patch_bbox, mip, T=T)
    self.pool.map(chunkwise, chunks)
    end = time()
    print (": {} sec".format(end - start))

  def multi_match(self, inverse=False, render=True):
    """Match current z to all tgt sections within tgt_radius
    Use to compare alignments of multiple sections to use consensus in 
    generating final alignment or masks for the section z.

    Args:
       inverse: bool indicating whether to align src to tgt or tgt to src
       render: bool indicating whether to render section

    Returns list of paths where fields were written
    """
    bbox = self.total_bbox
    for z_offset in range(1, self.tgt_radius):
      self.set_z_offset(z_offset)
      self.compute_section_pair_residuals(bbox, inverse=inverse)
      if render:
        self.render_section_all_mips(bbox)

  ## Whole stack operations
  def align_stack(self, z_range, bbox, move_anchor=True):
    if not self.check_all_params():
      raise Exception("Not all parameters are set")
    #  raise Exception("Have to align a chunkaligned size")

    self.total_bbox = bbox
    start = time()
    if move_anchor:
      self.set_z(z_range[0])
      for m in range(self.render_low_mip, self.high_mip+1):
        self.copy_section(bbox, mip=m)
      z_range = z_range[1:]
    for z in z_range:
      self.set_z(z)
      self.compute_section_pair_residuals(bbox)
      self.render_section_all_mips(bbox)
    end = time()
    print("Total time for aligning {} slices: {}".format(end_section-start_section,
                                                          end - start))

  def align_stack_vector_vote(self, z_range, bbox, render_multi_match=False):
    """Align stack of images using vector voting within tgt_radius 
  
    Args:
        z_range: list of z indices to be aligned 
        bbox: BoundingBox object for bounds of 2D region
        render_multi_match: Bool indicating whether to separately render out
            each aligned section before compiling vector fields with voting
            (useful for debugging)
    """
    aligner.total_bbox = bbox
    # start_z = z_range[0]
    # if start_without:
    #   aligner.align_stack(z_range, bbox, move_anchor=False) 
    #   z_range = z_range[3:]
    for z in z_range:
      aligner.multi_match(inverse=False, render=render_multi_match)
      # print('align_vector_vote field_paths: {0}'.format(field_paths))
      mip = aligner.process_low_mip
      T = 2**mip
      print('softmin temp: {0}'.format(T))
      aligner.compute_weighted_field(z, bbox, mip, T)
      aligner.render_section_all_mips(z, bbox, start_z)
  
