from process import Process
from cloudvolume import CloudVolume as cv
from cloudvolume.lib import Vec
import torch
import numpy as np
import os
from os.path import join
import json
from time import time
from copy import deepcopy, copy
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
               render_low_mip=2, render_high_mip=6, is_Xmas=False, threads=5,
               max_chunk=(1024, 1024), max_render_chunk=(2048*2, 2048*2),
               skip=0, topskip=0, size=7, should_contrast=True, num_targets=1,
               flip_average=True, write_intermediaries=False,
               upsample_residuals=False, old_upsample=False, old_vectors=False,
               ignore_field_init=False, z_offset=0):
    self.process_high_mip = mip_range[1]
    self.process_low_mip  = mip_range[0]
    self.render_low_mip   = render_low_mip
    self.render_high_mip  = render_high_mip
    self.high_mip         = max(self.render_high_mip, self.process_high_mip)
    self.high_mip_chunk   = high_mip_chunk
    self.max_chunk        = max_chunk
    self.max_render_chunk = max_render_chunk
    self.num_targets      = num_targets
    self.size = size
    self.old_vectors=old_vectors
    self.ignore_field_init = ignore_field_init
    self.z_offset = z_offset

    self.max_displacement = max_displacement
    self.crop_amount      = crop

    self.orig_src_path = src_path
    self.orig_tgt_path = tgt_path
    self.orig_dst_path = dst_path
    self.paths = self.get_paths(src_path, tgt_path, dst_path)

    self.net = Process(model_path, mip_range[0], is_Xmas=is_Xmas, cuda=True, dim=high_mip_chunk[0]+crop*2, skip=skip, topskip=topskip, size=size, flip_average=flip_average, old_upsample=old_upsample)
    
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

  def reset_paths(self):
    self.paths = self.get_paths(self.orig_src_path, self.orig_dst_path)

  def get_paths(self, src_path, tgt_path, dst_path):
    paths = {}
    paths['dst_path'] = dst_path
    paths['src_img'] = src_path
    paths['tgt_img'] = tgt_path
    paths['dst_img'] = join(dst_path, 'image')
    paths['tmp_img'] = join(dst_path, 'intermediate')
    mip_range = range(self.process_high_mip + 10) #TODO
    paths['enc'] = [join(dst_path, 'enc/{}'.format(i)) for i in mip_range]

    res   = [join(dst_path, 'vec/{}'.format(i)) for i in mip_range]
    paths['x_res'] = [join(r, 'x') for r in res]
    paths['y_res'] = [join(r, 'y') for r in res]

    cumres   = [join(dst_path, 'cumulative_vec/{}'.format(i)) 
                                                     for i in mip_range] 
    paths['x_cumres'] = [join(r, 'x') for r in cumres]
    paths['y_cumres'] = [join(r, 'y') for r in cumres]

    resup   = [join(dst_path, 'vec_up/{}'.format(i)) for i in mip_range]
    paths['x_resup'] = [join(r, 'x') for r in resup]
    paths['y_resup'] = [join(r, 'y') for r in resup]

    cumresup   = [join(dst_path, 'cumulative_vec_up/{}'.format(i)) 
                                                     for i in mip_range]
    paths['x_cumresup'] = [join(r, 'x') for r in cumresup]
    paths['y_cumresup'] = [join(r, 'y') for r in cumresup]

    field   = [join(dst_path, 'field/{}'.format(i)) for i in mip_range]
    paths['x_field'] = [join(r, 'x') for r in field]
    paths['y_field'] = [join(r, 'y') for r in field]

    return paths

  def set_chunk_size(self, chunk_size):
    self.high_mip_chunk = chunk_size

  def _create_info_files(self, max_offset):
    src_cv = cv(self.paths['src_img'])
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

    cv(self.paths['dst_img'], info=dst_info).commit_info()
    cv(self.paths['tmp_img'], info=dst_info).commit_info()

    ##########################################################
    #### Create vec info file
    ##########################################################
    vec_info = deepcopy(src_info)
    vec_info["data_type"] = "float32"
    for i in range(len(vec_info["scales"])):
      vec_info["scales"][i]["chunk_sizes"][0][2] = 1

    enc_dict = {x: 6*(x-self.process_low_mip)+12 for x in 
                    range(self.process_low_mip, self.process_high_mip+1)} 

    scales = deepcopy(vec_info["scales"])
    # print('src_info scales: {0}'.format(len(scales)))
    for i in range(len(scales)):
      self.vec_chunk_sizes.append(scales[i]["chunk_sizes"][0][0:2])
      self.vec_voxel_offsets.append(scales[i]["voxel_offset"])
      self.vec_total_sizes.append(scales[i]["size"])
      if not self.ignore_field_init:
        cv(self.paths['x_field'][i], info=vec_info).commit_info()
        cv(self.paths['y_field'][i], info=vec_info).commit_info()
      cv(self.paths['x_res'][i], info=vec_info).commit_info()
      cv(self.paths['y_res'][i], info=vec_info).commit_info()
      cv(self.paths['x_cumres'][i], info=vec_info).commit_info()
      cv(self.paths['y_cumres'][i], info=vec_info).commit_info()
      cv(self.paths['x_resup'][i], info=vec_info).commit_info()
      cv(self.paths['y_resup'][i], info=vec_info).commit_info()
      cv(self.paths['x_cumresup'][i], info=vec_info).commit_info()
      cv(self.paths['y_cumresup'][i], info=vec_info).commit_info()

      if i in enc_dict.keys():
        enc_info = deepcopy(vec_info)
        enc_info['num_channels'] = enc_dict[i]
        # enc_info['data_type'] = 'uint8'
        cv(self.paths['enc'][i], info=enc_info).commit_info()

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
                         mip=0, max_mip=self.process_high_mip)
    return result

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
                                 mip=mip, max_mip=self.high_mip))

    return chunks

  def compute_residual_patch(self, source_z, target_z, out_patch_bbox, mip):
    print ("Computing residual for region {}.".format(out_patch_bbox.__str__(mip=0)), flush=True)
    precrop_patch_bbox = deepcopy(out_patch_bbox)
    precrop_patch_bbox.uncrop(self.crop_amount, mip=mip)

    if mip == self.process_high_mip:
      src_patch = self.get_image_data(self.paths['src_img'], source_z, precrop_patch_bbox, mip)
    else:
      src_patch = self.get_image_data(self.paths['tmp_img'], source_z, precrop_patch_bbox, mip)

    tgt_patch = self.get_image_data(self.paths['tgt_img'], target_z, precrop_patch_bbox, mip, should_backtrack=True)
    field, residuals, encodings, cum_residuals = self.net.process(src_patch, tgt_patch, mip, crop=self.crop_amount, old_vectors=self.old_vectors)
    #rel_residual = precrop_patch_bbox.spoof_x_y_residual(1024, 0, mip=mip,
    #                        crop_amount=self.crop_amount)

    # save the final vector field for warping
    self.save_vector_patch(field, self.paths['x_field'][mip], self.paths['y_field'][mip], source_z, out_patch_bbox, mip)

    if self.write_intermediaries:
  
      mip_range = range(self.process_low_mip+self.size-1, self.process_low_mip-1, -1)
      for res_mip, res, cumres in zip(mip_range, residuals[1:], cum_residuals[1:]):
          crop = self.crop_amount // 2**(res_mip - self.process_low_mip)   
          self.save_residual_patch(res, crop, self.paths['x_res'][res_mip], 
                                   self.paths['y_res'][res_mip], source_z, 
                                   out_patch_bbox, res_mip)
          self.save_residual_patch(cumres, crop, self.paths['x_cumres'][res_mip], 
                                   self.paths['y_cumres'][res_mip], source_z, 
                                   out_patch_bbox, res_mip)
          if self.upsample_residuals:
            crop = self.crop_amount   
            res = self.scale_residuals(res, res_mip, self.process_low_mip)
            self.save_residual_patch(res, crop, self.paths['x_resup'][res_mip], 
                                     self.paths['y_resup'][res_mip], source_z, 
                                     out_patch_bbox, self.process_low_mip)
            cumres = self.scale_residuals(cumres, res_mip, self.process_low_mip)
            self.save_residual_patch(cumres, crop, self.paths['x_cumresup'][res_mip], 
                                     self.paths['y_cumresup'][res_mip], source_z, 
                                     out_patch_bbox, self.process_low_mip)


 
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
            cv(self.paths['enc'][mip], mip=mip, bounded=False, fill_missing=True, autocrop=True,
                                    progress=False)[x_range[0]:x_range[1],
                                                    y_range[0]:y_range[1], z, j_slice] = patch # uint_patch
  
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


  ## Patch manipulation
  def warp_patch(self, path, z, bbox, res_mip_range, mip):
    influence_bbox = deepcopy(bbox)
    influence_bbox.uncrop(self.max_displacement, mip=0)
    start = time()

    agg_flow = self.get_aggregate_rel_flow(z, influence_bbox, res_mip_range, mip)
    image = torch.from_numpy(self.get_image_data(path, z, influence_bbox, mip))
    image = image.unsqueeze(0)

    #no need to warp if flow is identity since warp introduces noise
    if torch.min(agg_flow) != 0 or torch.max(agg_flow) != 0:
      image = gridsample_residual(image, agg_flow, padding_mode='zeros')
    else:
      print ("not warping")

    mip_disp = int(self.max_displacement / 2**mip)
    return image.numpy()[0,:,mip_disp:-mip_disp,mip_disp:-mip_disp]

  def downsample_patch(self, path, z, bbox, mip):
    in_data = self.get_image_data(path, z, bbox, mip - 1)
    result = np_downsample(in_data, 2)
    return result

  ## Data saving
  def save_image_patch(self, path, float_patch, z, bbox, mip):
    x_range = bbox.x_range(mip=mip)
    y_range = bbox.y_range(mip=mip)
    patch = float_patch[0, :, :, np.newaxis]
    uint_patch = (np.multiply(patch, 255)).astype(np.uint8)
    cv(path, mip=mip, bounded=False, fill_missing=True, autocrop=True,
                                  progress=False)[x_range[0]:x_range[1],
                                                  y_range[0]:y_range[1], z] = uint_patch

  def scale_residuals(self, res, src_mip, dst_mip):
    print('Upsampling residuals from MIP {0} to {1}'.format(src_mip, dst_mip))
    up = nn.Upsample(scale_factor=2, mode='bilinear')
    for m in range(src_mip, dst_mip, -1):
      res = up(res.permute(0,3,1,2)).permute(0,2,3,1)
    return res

  def save_residual_patch(self, res, crop, x_path, y_path, z, bbox, mip):
    print ("Saving residual patch {} at MIP {}".format(bbox.__str__(mip=0), mip))
    v = res * (res.shape[-2] / 2) * (2**mip)
    v = v[:,crop:-crop, crop:-crop,:]
    v = v.data.cpu().numpy() 
    self.save_vector_patch(v, x_path, y_path, z, bbox, mip)

  def save_vector_patch(self, flow, x_path, y_path, z, bbox, mip):
    x_res = flow[0, :, :, 0, np.newaxis]
    y_res = flow[0, :, :, 1, np.newaxis]

    x_range = bbox.x_range(mip=mip)
    y_range = bbox.y_range(mip=mip)

    cv(x_path, mip=mip, bounded=False, fill_missing=True, autocrop=True,
                       non_aligned_writes=False, progress=False)[x_range[0]:x_range[1],
                                                   y_range[0]:y_range[1], z] = x_res
    cv(y_path, mip=mip, bounded=False, fill_missing=True, autocrop=True,
                       non_aligned_writes=False, progress=False)[x_range[0]:x_range[1],
                                                   y_range[0]:y_range[1], z] = y_res

  ## Data loading
  def preprocess_data(self, data):
    sd = np.squeeze(data)
    ed = np.expand_dims(sd, 0)
    nd = np.divide(ed, float(255.0), dtype=np.float32)
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

  def get_image_data(self, path, z, bbox, mip, should_backtrack=False):
    #data = self.check_image_cache(path, bbox, mip)
    #if data is not None:
    #  return data
    data = None
    x_range = bbox.x_range(mip=mip)
    y_range = bbox.y_range(mip=mip)
    while data is None:
      try:
        data_ = cv(path, mip=mip, progress=False,
                   bounded=False, fill_missing=True)[x_range[0]:x_range[1], y_range[0]:y_range[1], z]
        data = data_
      except AttributeError as e:
        pass
    
    if self.num_targets > 1 and should_backtrack:
      for backtrack in range(1, self.num_targets):
        if z-backtrack < self.zs:
          break
        still_missing_mask = self.missing_data_mask(data, bbox, mip)
        if not np.any(still_missing_mask):
          break # we've got a full slice
        backup = None
        while backup is None:
          try:
            backup_ = cv(path, mip=mip, progress=False,
                         bounded=False, fill_missing=True)[x_range[0]:x_range[1], y_range[0]:y_range[1], z-backtrack]
            backup = backup_
          except AttributeError as e:
            pass
          
        self.supplement_target_with_backup(data, still_missing_mask, backup, bbox, mip)
        
    data = self.preprocess_data(data)
    #self.add_to_image_cache(path, bbox, mip, data)

    return data

  def get_vector_data(self, path, z, bbox, mip):
    x_range = bbox.x_range(mip=mip)
    y_range = bbox.y_range(mip=mip)

    data = None
    while data is None:
      try:
        data_ = cv(path, mip=mip, progress=False,
                   bounded=False, fill_missing=True)[x_range[0]:x_range[1], y_range[0]:y_range[1], z]
        data = data_
      except AttributeError as e:
        pass
    return data

  def get_abs_residual(self, z, bbox, mip):
    x = self.get_vector_data(self.paths['x_field'][mip], z, bbox, mip)[..., 0, 0]
    y = self.get_vector_data(self.paths['y_field'][mip], z, bbox, mip)[..., 0, 0]
    result = np.stack((x, y), axis=2)
    return np.expand_dims(result, axis=0)

  def get_rel_residual(self, z, bbox, mip):
    x = self.get_vector_data(self.paths['x_field'][mip], z, bbox, mip)[..., 0, 0]
    y = self.get_vector_data(self.paths['y_field'][mip], z, bbox, mip)[..., 0, 0]
    abs_res = np.stack((x, y), axis=2)
    abs_res = np.expand_dims(abs_res, axis=0)
    rel_res = self.abs_to_rel_residual(abs_res, bbox, mip)
    return rel_res


  def get_aggregate_rel_flow(self, z, bbox, res_mip_range, mip):
    result = torch.zeros((1, bbox.x_size(mip), bbox.y_size(mip), 2), dtype=torch.float)
    start_mip = max(res_mip_range[0], self.process_low_mip)
    end_mip   = min(res_mip_range[1], self.process_high_mip)

    for res_mip in range(start_mip, end_mip + 1):
      rel_res = torch.from_numpy(self.get_rel_residual(z, bbox, res_mip))
      up_rel_res = upsample(res_mip - mip)(rel_res.permute(0,3,1,2)).permute(0,2,3,1)
      result += up_rel_res

    return result

  ## High level services
  def copy_section(self, source, dest, z, bbox, mip):
    print ("moving section {} mip {} to dest".format(z, mip), end='', flush=True)
    start = time()
    chunks = self.break_into_chunks(bbox, self.dst_chunk_sizes[mip],
                                    self.dst_voxel_offsets[mip], mip=mip, render=True)
    #for patch_bbox in chunks:
    def chunkwise(patch_bbox):
      raw_patch = self.get_image_data(source, z, patch_bbox, mip)
      self.save_image_patch(dest, raw_patch, z, patch_bbox, mip)

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

  def render(self, z, bbox, mip):
    print ("Rendering mip {}".format(mip),
              end='', flush=True)
    start = time()
    chunks = self.break_into_chunks(bbox, self.dst_chunk_sizes[mip],
                                    self.dst_voxel_offsets[mip], mip=mip, render=True)

    def chunkwise(patch_bbox):
      warped_patch = self.warp_patch(self.paths['src_img'], z, patch_bbox,
                                    (mip, self.process_high_mip), mip)
      self.save_image_patch(self.paths['dst_img'], warped_patch, z+self.z_offset, patch_bbox, mip)
    self.pool.map(chunkwise, chunks)
    end = time()
    print (": {} sec".format(end - start))

  def render_section_all_mips(self, z, bbox):
    self.render(z, bbox, self.render_low_mip)
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

  ## Whole stack operations
  def align_ng_stack(self, start_section, end_section, bbox, move_anchor=True):
    if not self.check_all_params():
      raise Exception("Not all parameters are set")
    #  raise Exception("Have to align a chunkaligned size")

    self.total_bbox = bbox
    
    start = time()
    if move_anchor:
      for m in range(self.render_low_mip, self.high_mip+1):
        self.copy_section(self.paths['src_img'], self.paths['dst_img'], start_section, bbox, mip=m)
    self.zs = start_section
    for z in range(start_section, end_section):
      self.img_cache = {}
      self.compute_section_pair_residuals(z + 1, z, bbox)
      self.render_section_all_mips(z + 1, bbox)
    end = time()
    print ("Total time for aligning {} slices: {}".format(end_section - start_section,
                                                          end - start))

  def multi_match(self, tgt_z, src_z_list):
    """Match series of sources to single target section. Primary purpose is to
    compare alignments of multiple sections to use consensus in generating
    masks for the target section.

    Args:
       tgt_z: z index of target section
       src_z_list: list of z indices for source sections
    """
    orig_src_path = self.orig_src_path
    orig_tgt_path = self.orig_tgt_path
    orig_dst_path = self.orig_dst_path 
    bbox = self.total_bbox
    for src_z in src_z_list:
      print('Aligning {0} to {1}'.format(src_z, tgt_z))
      self.zs = src_z
      z_offset = tgt_z - src_z
      self.z_offset = z_offset
      dst_path = '{0}/z{1}'.format(orig_dst_path, z_offset)
      self.paths = self.get_paths(orig_src_path, orig_tgt_path, dst_path)
      self._create_info_files(self.max_displacement)
      self.compute_section_pair_residuals(src_z, tgt_z, bbox)
      print('Rendering to {0}'.format(self.paths['dst_img'])) 
      self.render_section_all_mips(src_z, bbox)

