from process import Process
from cloudvolume import CloudVolume as cv
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

class Aligner:
  def __init__(self, model_path, max_displacement, crop,
               mip_range, high_mip_chunk, src_path, tgt_path, dst_path,
               src_mask_path='', src_mask_mip=0, src_mask_val=1, 
               tgt_mask_path='', tgt_mask_mip=0, tgt_mask_val=1,
               disable_cuda=False, max_mip=12,
               render_low_mip=2, render_high_mip=6, is_Xmas=False, threads=5,
               max_chunk=(1024, 1024), max_render_chunk=(2048*2, 2048*2),
               skip=0, topskip=0, size=7, should_contrast=True, 
               run_pairs=True, disable_flip_average=False, write_intermediaries=False,
               upsample_residuals=False, old_upsample=False, old_vectors=False,
               ignore_field_init=False, z_offset=0, **kwargs):
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
    self.orig_z_offset = z_offset
    self.z_offset = z_offset

    self.max_displacement = max_displacement
    self.crop_amount      = crop
    self.disable_cuda = disable_cuda
    self.device = torch.device('cpu') if disable_cuda else torch.device('cuda')

    self.orig_src_path = src_path
    self.orig_tgt_path = tgt_path
    self.orig_dst_path = dst_path
    self.src_mask_path = src_mask_path
    self.src_mask_mip  = src_mask_mip
    self.src_mask_val  = src_mask_val
    self.tgt_mask_path = tgt_mask_path
    self.tgt_mask_mip  = tgt_mask_mip
    self.tgt_mask_val  = tgt_mask_val
    self.paths = self.get_paths(src_path, tgt_path, dst_path, 
                                          src_mask_path, tgt_mask_path)

    self.net = Process(model_path, mip_range[0], is_Xmas=is_Xmas, cuda=True, dim=high_mip_chunk[0]+crop*2, skip=skip, topskip=topskip, size=size, flip_average=not disable_flip_average, old_upsample=old_upsample)

    self.normalizer = Normalizer(min(5, mip_range[0])) 
    self.write_intermediaries = write_intermediaries
    self.upsample_residuals = upsample_residuals

    self.dst_chunk_sizes   = []
    self.dst_voxel_offsets = []
    self.vec_chunk_sizes   = []
    self.vec_voxel_offsets = []
    self.vec_total_sizes   = []
    self._create_info_files(max_displacement)
    self.pool = ThreadPool(threads)

    self.img_cache = {}

    self.img_cache_lock = Lock()

  def reset(self):
    self.z_offset = self.orig_z_offset
    self.paths = self.get_paths(self.orig_src_path, self.orig_tgt_path, 
                                self.orig_dst_path, self.src_mask_path, 
                                self.tgt_mask_path)

  def get_paths(self, src_path, tgt_path, dst_path, 
                                src_mask_path='', tgt_mask_path=''):
    paths = {}
    paths['dst_path'] = dst_path
    paths['src_img'] = src_path
    paths['src_mask'] = src_mask_path
    paths['tgt_mask'] = tgt_mask_path
    paths['tgt_img'] = tgt_path
    paths['dst_img'] = join(dst_path, 'image')
    paths['tmp_img'] = join(dst_path, 'intermediate')
    mip_range = range(self.process_high_mip + 10) #TODO
    paths['enc'] = [join(dst_path, 'enc/{}'.format(i)) for i in mip_range]

    paths['res'] = [join(dst_path, 'vec/{}'.format(i)) for i in mip_range]
    paths['cumres'] = [join(dst_path, 'cumulative_vec/{}'.format(i)) 
                                                     for i in mip_range] 
    paths['resup'] = [join(dst_path, 'vec_up/{}'.format(i)) for i in mip_range]
    paths['cumresup'] = [join(dst_path, 'cumulative_vec_up/{}'.format(i)) 
                                                     for i in mip_range]
    paths['field']   = [join(dst_path, 'field/{}'.format(i)) for i in mip_range]
    paths['field_sf'] = [join(dst_path, 'field_sf'.format(i)) for i in mip_range]
    paths['diffs']   = [join(dst_path, 'diffs'.format(i)) for i in mip_range]
    paths['diff_weights']   = [join(dst_path, 'diff_weights'.format(i)) for i in mip_range]
    paths['weights']   = [join(dst_path, 'weights'.format(i)) for i in mip_range]

    return paths

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

  def _create_info_files(self, max_offset):
    src_cv = cv(self.paths['src_img'], provenance={})
    src_info = src_cv.info
    m = len(src_info['scales'])
    each_factor = Vec(2,2,1)
    factor = Vec(2**m,2**m,1)
    for _ in range(m, self.process_low_mip + self.size):
      src_cv.add_scale(factor)
      factor *= each_factor
      chunksize = src_info['scales'][-2]['chunk_sizes'][0] // each_factor
      src_info['scales'][-1]['chunk_sizes'] = [ list(map(int, chunksize)) ]

    # print(src_info)
    dst_info = deepcopy(src_info)

    ##########################################################
    #### Create dst info file
    ##########################################################
    chunk_size = dst_info["scales"][0]["chunk_sizes"][0][0]
    dst_size_increase = max_offset
    if dst_size_increase % chunk_size != 0:
      dst_size_increase = dst_size_increase - (dst_size_increase % max_offset) + chunk_size
    scales = dst_info["scales"]
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

    cv(self.paths['dst_img'], info=dst_info, provenance={}).commit_info()
    cv(self.paths['tmp_img'], info=dst_info, provenance={}).commit_info()

    ##########################################################
    #### Create vec info file
    ##########################################################
    vec_info = deepcopy(dst_info)
    vec_info["data_type"] = "float32"
    for i in range(len(vec_info["scales"])):
      vec_info["scales"][i]["chunk_sizes"][0][2] = 1
    vec_info['num_channels'] = 2

    enc_dict = {x: 6*(x-self.process_low_mip)+12 for x in 
                    range(self.process_low_mip, self.process_high_mip+1)} 

    scales = deepcopy(vec_info["scales"])

    for i in range(len(scales)):
      self.vec_chunk_sizes.append(scales[i]["chunk_sizes"][0][0:2])
      self.vec_voxel_offsets.append(scales[i]["voxel_offset"])
      self.vec_total_sizes.append(scales[i]["size"])
      if not self.ignore_field_init:
        cv(self.paths['field'][i], info=vec_info, provenance={}).commit_info()
      cv(self.paths['field_sf'][i], info=vec_info, provenance={}).commit_info() 
      cv(self.paths['res'][i], info=vec_info, provenance={}).commit_info()
      cv(self.paths['cumres'][i], info=vec_info, provenance={}).commit_info()
      cv(self.paths['resup'][i], info=vec_info, provenance={}).commit_info()
      cv(self.paths['cumresup'][i], info=vec_info, provenance={}).commit_info()

      if i in enc_dict.keys():
        enc_info = deepcopy(vec_info)
        enc_info['num_channels'] = enc_dict[i]
        # enc_info['data_type'] = 'uint8'
        cv(self.paths['enc'][i], info=enc_info, provenance={}).commit_info()

      wts_info = deepcopy(vec_info)
      wts_info['num_channels'] = 3
      # enc_info['data_type'] = 'uint8'
      cv(self.paths['diffs'][i], info=wts_info, provenance={}).commit_info()
      cv(self.paths['diff_weights'][i], info=wts_info, provenance={}).commit_info()
      cv(self.paths['weights'][i], info=wts_info, provenance={}).commit_info()
    

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

  def get_field(self, path, z, bbox, mip, relative=False, to_tensor=True):
    x_range = bbox.x_range(mip=mip)
    y_range = bbox.y_range(mip=mip)
    field = cv(path[mip], mip=mip, bounded=False, 
                  fill_missing=True, progress=False)[x_range[0]:x_range[1], 
                                                     y_range[0]:y_range[1], z]
    res = np.expand_dims(np.squeeze(field), axis=0)
    if relative:
      res = self.abs_to_rel_residual(res, bbox, mip)
    if to_tensor:
      res = torch.from_numpy(res)
      return res.to(device=self.device)
    else:
      return res

  def save_vector_patch(self, path, field, z, bbox, mip):
    x_range = bbox.x_range(mip=mip)
    y_range = bbox.y_range(mip=mip)
    field = np.squeeze(field)[:, :, np.newaxis, :]
    cv(path, mip=mip, bounded=False, fill_missing=True, autocrop=True,
       progress=False)[x_range[0]:x_range[1], y_range[0]:y_range[1], z] = field

  def save_residual_patch(self, path, res, crop, z, bbox, mip):
    print ("Saving residual patch {} at MIP {}".format(bbox.__str__(mip=0), mip))
    v = res * (res.shape[-2] / 2) * (2**mip)
    v = v[:,crop:-crop, crop:-crop,:]
    v = v.data.cpu().numpy() 
    self.save_field_patch(path, v, z, bbox, mip)

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

  def weight_fields(self, field_paths, z, bbox, mip, T=1, write_intermediaries=False):
    fields = [self.get_field(path, z, bbox, mip, relative=False) for path in field_paths]
    # field = vector_vote(fields, T=T)
    diffs = get_diffs(fields)
    diff_weights = weight_diffs(diffs, T=T)
    field_weights = compile_field_weights(diff_weights)
    field = weighted_sum_fields(field_weights, fields)
    self.save_vector_patch(self.paths['field'][mip], field, z, bbox, mip)

    if write_intermediaries:
      self.save_image_patch(self.paths['diffs'][mip], 
                            diffs.cpu().numpy(), z, bbox, mip, to_uint8=False)
      self.save_image_patch(self.paths['diff_weights'][mip], 
                            diff_weights.cpu().numpy(), z, bbox, mip, to_uint8=False)
      self.save_image_patch(self.paths['weights'][mip], 
                            field_weights.cpu().numpy(), z, bbox, mip, to_uint8=False)

  def compute_residual_patch(self, source_z, target_z, out_patch_bbox, mip):
    print ("Computing residual for region {}.".format(out_patch_bbox.__str__(mip=0)), flush=True)
    precrop_patch_bbox = deepcopy(out_patch_bbox)
    precrop_patch_bbox.uncrop(self.crop_amount, mip=mip)

    if mip == self.process_high_mip:
      src_patch = self.get_image(self.paths['src_img'], source_z, 
                                  precrop_patch_bbox, mip,
                                  adjust_contrast=True, to_tensor=True)
    else:
      src_patch = self.get_image(self.paths['tmp_img'], source_z, 
                                  precrop_patch_bbox, mip,
                                  adjust_contrast=True, to_tensor=True)

    tgt_patch = self.get_image(self.paths['tgt_img'], target_z, 
                                precrop_patch_bbox, mip,
                                adjust_contrast=True, to_tensor=True) 

    if self.paths['src_mask']:
      src_mask = self.get_mask(self.paths['src_mask'], source_z, 
                                precrop_patch_bbox, src_mip=self.src_mask_mip,
                                dst_mip=mip, valid_val=self.src_mask_val)
      src_patch = src_patch.masked_fill_(src_mask, 0)
    if self.paths['tgt_mask']:
      tgt_mask = self.get_mask(self.paths['tgt_mask'], target_z, 
                                precrop_patch_bbox, src_mip=self.tgt_mask_mip,
                                dst_mip=mip, valid_val=self.tgt_mask_val)
      tgt_patch = tgt_patch.masked_fill_(tgt_mask, 0)

    X = self.net.process(src_patch, tgt_patch, mip, crop=self.crop_amount, 
                                                 old_vectors=self.old_vectors)
    field, residuals, encodings, cum_residuals = X

    # save the final vector field for warping
    self.save_vector_patch(self.paths['field'][mip], field, source_z  out_patch_bbox, mip)

    if self.write_intermediaries:
      mip_range = range(self.process_low_mip+self.size-1, self.process_low_mip-1, -1)
      for res_mip, res, cumres in zip(mip_range, residuals[1:], cum_residuals[1:]):
          crop = self.crop_amount // 2**(res_mip - self.process_low_mip)   
          self.save_residual_patch(self.paths['res'][res_mip], res, crop, 
                                   source_z, out_patch_bbox, res_mip)
          self.save_residual_patch(self.paths['cumres'][res_mip], cumres, crop, 
                                   source_z, out_patch_bbox, res_mip)
          if self.upsample_residuals:
            crop = self.crop_amount   
            res = self.scale_residuals(res, res_mip, self.process_low_mip)
            self.save_residual_patch(self.paths['resup'][self.process_low_mip], res, crop, 
                                     source_z, out_patch_bbox, 
                                     self.process_low_mip)
            cumres = self.scale_residuals(cumres, res_mip, self.process_low_mip)
            self.save_residual_patch(self.paths['cumresup'][self.process_low_mip],
                                     cumres, crop, 
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

    x_chunk = self.dst_chunk_sizes[mip][0]
    y_chunk = self.dst_chunk_sizes[mip][1]

    x_offset = self.dst_voxel_offsets[mip][0]
    y_offset = self.dst_voxel_offsets[mip][1]

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
  def warp_patch(self, ng_path, z, bbox, res_mip_range, mip, start_z=-1):
    influence_bbox = deepcopy(bbox)
    influence_bbox.uncrop(self.max_displacement, mip=0)
    start = time()
    
    agg_flow = self.get_aggregate_rel_flow(z, influence_bbox, res_mip_range, mip)
    mip_disp = int(self.max_displacement / 2**mip)
    # image = torch.from_numpy(self.get_image(path, z, influence_bbox, mip))
    # image = image.unsqueeze(0)
    image = self.get_image(path, z, influence_bbox, mip, 
                                 adjust_contrast=False, to_tensor=True)
    if self.paths['src_mask']:
      mask = self.get_mask(self.paths['src_mask'], z, 
                                  influence_bbox, src_mip=self.src_mask_mip,
                                  dst_mip=mip, valid_val=self.src_mask_val)
      image = image.masked_fill_(mask, 0)

    # print('warp_patch shape {0}'.format(image.shape))
    # no need to warp if flow is identity since warp introduces noise
    if torch.min(agg_flow) != 0 or torch.max(agg_flow) != 0:
      image = gridsample_residual(image, agg_flow, padding_mode='zeros')
    else:
      print ("not warping")
    if (self.run_pairs):
      #cid = self.get_bbox_id(bbox, mip) 
      #print ("cid is ", cid)
      decay_factor = 0.4
      if z != start_z:
        field_sf = self.get_field(self.paths['field_sf'], z-1, influence_bbox, mip, relative=True, to_tensor=True)
        regular_part_x = torch.from_numpy(scipy.ndimage.filters.gaussian_filter((field_sf[...,0]), 256)).unsqueeze(-1)
        regular_part_y = torch.from_numpy(scipy.ndimage.filters.gaussian_filter((field_sf[...,1]), 256)).unsqueeze(-1)
        #regular_part = self.gauss_filter(field_sf.permute(3,0,1,2))
        #regular_part = torch.from_numpy(self.reg_field) 
        #field_sf = decay_factor * field_sf + (1 - decay_factor) * regular_part.permute(1,2,3,0) 
        #field_sf = regular_part.permute(1,2,3,0) 
        field_sf = torch.cat([regular_part_x,regular_part_y],-1)
        image = gridsample_residual(image, field_sf, padding_mode='zeros')
        agg_flow = agg_flow.permute(0,3,1,2)
        field_sf = field_sf + gridsample_residual(
            agg_flow, field_sf, padding_mode='border').permute(0,2,3,1)
      else:
        field_sf = agg_flow
      #self.calc_image_mean_field(image.numpy()[0,0,mip_disp:-mip_disp,mip_disp:-mip_disp], field_sf[0, mip_disp:-mip_disp, mip_disp:-mip_disp, :], cid)
      field_sf = field_sf * (field_sf.shape[-2] / 2) * (2**mip)
      # field_sf = field_sf.numpy()[:, mip_disp:-mip_disp, mip_disp:-mip_disp, :]
      self.save_residual_patch(self.paths['field_sf'][mip], field_sf, mip_disp, z, bbox, mip):

    if self.disable_cuda:
      return image.numpy()[:,:,mip_disp:-mip_disp,mip_disp:-mip_disp]
    else:
      return image.cpu().numpy()[:,:,mip_disp:-mip_disp,mip_disp:-mip_disp]

  def downsample_patch(self, path, z, bbox, mip):
    data = self.get_image(path, z, bbox, mip - 1, 
                              adjust_contrast=False, to_tensor=True)
    data = interpolate(data, scale_factor=0.5, mode='bilinear')
    return data.cpu().numpy()

  ## Data saving
  def save_image_patch(self, path, float_patch, z, bbox, mip, to_uint8=True):
    x_range = bbox.x_range(mip=mip)
    y_range = bbox.y_range(mip=mip)
    patch = np.transpose(float_patch, (3,2,1,0))
    if to_uint8:
      patch = (np.multiply(patch, 255)).astype(np.uint8)
    cv(path, mip=mip, bounded=False, fill_missing=True, autocrop=True,
                        cdn_cache=False, progress=False, provenance={})[x_range[0]:x_range[1],
                                                  y_range[0]:y_range[1], z] = patch

  def scale_residuals(self, res, src_mip, dst_mip):
    print('Upsampling residuals from MIP {0} to {1}'.format(src_mip, dst_mip))
    up = nn.Upsample(scale_factor=2, mode='bilinear')
    for m in range(src_mip, dst_mip, -1):
      res = up(res.permute(0,3,1,2)).permute(0,2,3,1)
    return res

  ## Data loading
  def preprocess_data(self, data, to_float=True, adjust_contrast=False, to_tensor=True):
    return nd

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
    
  def supplement_target_with_backup(self, target, still_missing_mask, backup, bbox, mip):
    backup_missing_mask = self.missing_data_mask(backup, bbox, mip)
    fill_in = backup_missing_mask < still_missing_mask
    target[fill_in] = backup[fill_in]

  def check_image_cache(self, path, bbox, mip):
    with self.img_cache_lock:
      output = -1 * np.ones((1,1,bbox.x_size(mip), bbox.y_size(mip)))
      for key in self.img_cache:
        other_path, other_bbox, other_mip = key[0], key[1], key[2]
        if other_mip == mip and other_path == path:
          if bbox.intersects(other_bbox):
            xs, ys, xsz, ysz = other_bbox.insets(bbox, mip)
            output[:,:,xs:xs+xsz,ys:ys+ysz] = self.img_cache[key]
    if np.min(output > -1):
      print('hit')
      return output
    else:
      return None

  def add_to_image_cache(self, path, bbox, mip, data):
    with self.img_cache_lock:
      self.img_cache[(path, bbox, mip)] = data

  def get_mask(self, path, z, bbox, src_mip, dst_mip, valid_val, to_tensor=True):
    data = self.get_data(path, z, bbox, src_mip=src_mip, dst_mip=dst_mip, 
                             to_float=False, adjust_contrast=False, 
                             to_tensor=to_tensor)
    return data == valid_val

  def get_image(self, path, z, bbox, mip, adjust_contrast=False, to_tensor=True):
    return self.get_data(path, z, bbox, src_mip=mip, dst_mip=mip, to_float=True, 
                             adjust_contrast=adjust_contrast, to_tensor=to_tensor)

  def get_data(self, path, z, bbox, src_mip, dst_mip, 
                     to_float=True, adjust_contrast=False, to_tensor=True):
    """Retrieve CloudVolume data. Returns 4D ndarray or tensor, BxCxWxH
    
    Args:
       path: CloudVolume path
       z: z index
       bbox: BoundingBox defining data range
       src_mip: mip of the CloudVolume data
       dst_mip: mip of the output mask (dictates whether to up/downsample)
       to_float: output should be float32
       adjust_contrast: output will be normalized
       to_tensor: output will be torch.tensor
    """
    x_range = bbox.x_range(mip=src_mip)
    y_range = bbox.y_range(mip=src_mip)
    data = cv(path, mip=src_mip, progress=False, bounded=False, 
             fill_missing=True, provenance={})[x_range[0]:x_range[1], y_range[0]:y_range[1], z] 
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

  def get_vector_data(self, path, z, bbox, mip):
    x_range = bbox.x_range(mip=mip)
    y_range = bbox.y_range(mip=mip)

    data = None
    while data is None:
      try:
        data_ = cv(path, mip=mip, progress=False,
                   bounded=False, fill_missing=True, provenance={})[x_range[0]:x_range[1], y_range[0]:y_range[1], z]
        data = data_
      except AttributeError as e:
        pass
    return data

  def get_aggregate_rel_flow(self, z, bbox, res_mip_range, mip):
    result = torch.zeros((1, bbox.x_size(mip), bbox.y_size(mip), 2), 
                                 dtype=torch.float, device=self.device)
    start_mip = max(res_mip_range[0], self.process_low_mip)
    end_mip   = min(res_mip_range[1], self.process_high_mip)

    for res_mip in range(start_mip, end_mip + 1):
      rel_res = torch.from_numpy(self.get_rel_residual(z, bbox, res_mip))
      rel_res = rel_res.to(device=self.device)
      up_rel_res = upsample(res_mip - mip)(rel_res.permute(0,3,1,2)).permute(0,2,3,1)
      result += up_rel_res

    return result

  ## High level services
  def copy_section(self, src_path, dst_path, z, bbox, mip):
    print ("moving section {} mip {} to dest".format(z, mip), end='', flush=True)
    start = time()
    chunks = self.break_into_chunks(bbox, self.dst_chunk_sizes[mip],
                                    self.dst_voxel_offsets[mip], mip=mip, render=True)
    #for patch_bbox in chunks:
    def chunkwise(patch_bbox):

      if self.paths['src_mask']:
        raw_patch = self.get_image(src_path, z, 
                                    patch_bbox, mip,
                                    adjust_contrast=False, to_tensor=True)
        raw_mask = self.get_mask(self.paths['src_mask'], z, 
                                 precrop_patch_bbox, src_mip=self.src_mask_mip,
                                 dst_mip=mip, valid_val=self.src_mask_val)
        raw_patch = raw_patch.masked_fill_(raw_mask, 0)
        raw_patch = raw_patch.cpu().numpy()
      else: 
        raw_patch = self.get_image(src_path, z, 
                                    patch_bbox, mip,
                                    adjust_contrast=False, to_tensor=False)
      self.save_image_patch(dst_path, raw_patch, z, patch_bbox, mip)

    self.pool.map(chunkwise, chunks)

    end = time()
    print (": {} sec".format(end - start))

  def prepare_source(self, z, bbox, mip):
    print ("Prerendering mip {}".format(mip),
           end='', flush=True)
    start = time()

    chunks = self.break_into_chunks(bbox, self.dst_chunk_sizes[mip],
                                    self.dst_voxel_offsets[mip], mip=mip, render=True)

    def chunkwise(patch_bbox):
      warped_patch = self.warp_patch(self.paths['src_img'], z, patch_bbox,
                                     (mip + 1, self.process_high_mip), mip)
      self.save_image_patch(self.paths['tmp_img'], warped_patch, z, patch_bbox, mip)
    self.pool.map(chunkwise, chunks)
    end = time()
    print (": {} sec".format(end - start))

  def render(self, z, bbox, mip, start_z):
    print('Rendering z={0} with z_offset={1} @ MIP{2}'.format(z, mip, self.z_offset), flush=True)
    print('src_path {0}'.format(self.paths['src_img']))
    print('dst_path {0}'.format(self.paths['dst_img']))
    start = time()
    chunks = self.break_into_chunks(bbox, self.dst_chunk_sizes[mip],
                                    self.dst_voxel_offsets[mip], mip=mip, render=True)
    #print("\n total chunsk is ", len(chunks))
    #if (self.run_pairs and (z!=start_z)):
    #    total_chunks = len(chunks) 
    #    self.reg_field= np.sum(self.field_sf_sum, axis=0) / np.sum(self.image_pixels_sum)
    #    self.image_pixels_sum = np.zeros(total_chunks)
    #    self.field_sf_sum = np.zeros((total_chunks, 2), dtype=np.float32)

    def chunkwise(patch_bbox):
      warped_patch = self.warp_patch(self.paths['src_img'], z, patch_bbox,
                                    (mip, self.process_high_mip), mip, start_z)
      self.save_image_patch(self.paths['dst_img'], warped_patch, z+self.z_offset, patch_bbox, mip)
    self.pool.map(chunkwise, chunks)
    end = time()
    print (": {} sec".format(end - start))

  def render_section_all_mips(self, z, bbox, start_z):
    self.render(z, bbox, self.render_low_mip, start_z)
    self.downsample(z, bbox, self.render_low_mip, self.render_high_mip)

  def downsample(self, z, bbox, source_mip, target_mip):
    print ("Downsampling {} from mip {} to mip {}".format(bbox.__str__(mip=0), source_mip, target_mip))
    for m in range(source_mip+1, target_mip + 1):
      chunks = self.break_into_chunks(bbox, self.dst_chunk_sizes[m],
                                      self.dst_voxel_offsets[m], mip=m, render=True)

      def chunkwise(patch_bbox):
        print ("Downsampling {} to mip {}".format(patch_bbox.__str__(mip=0), m))
        downsampled_patch = self.downsample_patch(self.paths['dst_img'], z+self.z_offset, patch_bbox, m)
        self.save_image_patch(self.paths['dst_img'], downsampled_patch, z+self.z_offset, patch_bbox, m)
      self.pool.map(chunkwise, chunks)

  def compute_section_pair_residuals(self, source_z, target_z, bbox):
    for m in range(self.process_high_mip,  self.process_low_mip - 1, -1):
      start = time()
      chunks = self.break_into_chunks(bbox, self.vec_chunk_sizes[m],
                                      self.vec_voxel_offsets[m], mip=m)
      print ("Aligning slice {} to slice {} at mip {} ({} chunks)".
             format(source_z, target_z, m, len(chunks)), flush=True)

      #for patch_bbox in chunks:
      def chunkwise(patch_bbox):
      #FIXME Torch runs out of memory
      #FIXME batchify download and upload
        self.compute_residual_patch(source_z, target_z, patch_bbox, mip=m)
      self.pool.map(chunkwise, chunks)
      end = time()
      print (": {} sec".format(end - start))

      if m > self.process_low_mip:
          self.prepare_source(source_z, bbox, m - 1)
    
  def count_box(self, bbox, mip):    
    chunks = self.break_into_chunks(bbox, self.dst_chunk_sizes[mip],
                                      self.dst_voxel_offsets[mip], mip=mip, render=True)
    total_chunks = len(chunks)
    self.image_pixels_sum =np.zeros(total_chunks)
    self.field_sf_sum =np.zeros((total_chunks, 2), dtype=np.float32)


  def compute_weighted_field(self, field_paths, z, bbox, mip, T=-1):
    """Chunked-processing of field weighting
    
    Args:
       field_paths: list of paths with field CloudVolumes
       z: section of fields to weight
       bbox: boundingbox of region to process
       mip: field MIP level
       T: softmin temperature (default will be 2**mip)
    """
    start = time()
    chunks = self.break_into_chunks(bbox, self.vec_chunk_sizes[mip],
                                    self.vec_voxel_offsets[mip], mip=mip)
    print("Computing weighted field for slice {0} @ MIP{1} ({2} chunks)".
           format(z, mip, len(chunks)), flush=True)
    print('Writing vectors to {0}'.format(self.paths['x_field'][mip]))

    #for patch_bbox in chunks:
    def chunkwise(patch_bbox):
      self.weight_fields(field_paths, z, patch_bbox, mip, T=T)
    self.pool.map(chunkwise, chunks)
    end = time()
    print (": {} sec".format(end - start))

  ## Whole stack operations
  def align_ng_stack(self, start_section, end_section, bbox, move_anchor=True):
    if not self.check_all_params():
      raise Exception("Not all parameters are set")
    #  raise Exception("Have to align a chunkaligned size")

    self.total_bbox = bbox
    start_z = start_section 
    start = time()
    if move_anchor:
      for m in range(self.render_low_mip, self.high_mip+1):
        self.copy_section(self.paths['src_img'], self.paths['dst_img'], start_section, bbox, mip=m)
        start_z = start_section + 1 
    self.zs = start_section
    for z in range(start_section, end_section):
      self.img_cache = {}
      self.compute_section_pair_residuals(z + 1, z, bbox)
      self.render_section_all_mips(z + 1, bbox, start_z)
    end = time()
    print ("Total time for aligning {} slices: {}".format(end_section - start_section,
                                                          end - start))

  def multi_match(self, z, z_list, inverse=False, render=True):
    """Match single section z to series of sections in z_list.
    If inverse=True, then match series of sections in z_list to section z.
    Use to compare alignments of multiple sections to use consensus in 
    generating final alignment or masks for the section z.

    Args:
       z: z index of single section
       z_list: list of z indices to match
       inverse: bool indicating whether to align z_list to z, 
         instead of z_list to z
       render: bool indicating whether to render section

    Returns list of paths where fields were written
    """
    orig_src_path = self.orig_src_path
    orig_tgt_path = self.orig_tgt_path
    orig_dst_path = self.orig_dst_path 
    bbox = self.total_bbox
    field_paths = []
    for tgt_z in z_list:
      src_z = z
      if inverse:
        src_z, tgt_z = tgt_z, src_z
      print('Aligning {0} to {1}'.format(src_z, tgt_z))
      self.zs = src_z
      z_offset = tgt_z - src_z
      self.z_offset = z_offset
      dst_path = '{0}/z{1}'.format(orig_dst_path, z_offset)
      field_paths.append(dst_path)
      self.paths = self.get_paths(orig_src_path, orig_tgt_path, dst_path)
      self._create_info_files(self.max_displacement)
      self.compute_section_pair_residuals(src_z, tgt_z, bbox)
      if render:
        print('Rendering to {0}'.format(self.paths['dst_img'])) 
        self.render_section_all_mips(src_z, bbox)
    return field_paths

