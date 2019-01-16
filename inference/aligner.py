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
from utilities.helpers import save_chunk, crop, upsample, gridsample_residual, \
                              np_downsample, invert

from skimage.morphology import disk as skdisk
from skimage.filters.rank import maximum as skmaximum 
from boundingbox import BoundingBox, deserialize_bbox

from pathos.multiprocessing import ProcessPool, ThreadPool
from threading import Lock

import torch.nn as nn
from directory_manager import SrcDir, DstDir

from task_handler import TaskHandler, make_residual_task_message, \
        make_render_task_message, make_copy_task_message, \
        make_downsample_task_message, make_compose_task_message, \
        make_prepare_task_message, make_vector_vote_task_message, \
        make_regularize_task_message, make_render_low_mip_task_message, \
        make_invert_field_task_message, make_render_cv_task_message

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
               render_low_mip=2, render_high_mip=9, is_Xmas=False, threads=1,
               max_chunk=(1024, 1024), max_render_chunk=(2048*2, 2048*2),
               skip=0, topskip=0, size=7, should_contrast=True, 
               disable_flip_average=False, write_intermediaries=False,
               upsample_residuals=False, old_upsample=False, old_vectors=False,
               ignore_field_init=False, z=0, tgt_radius=1, 
               queue_name=None, p_render=False, dir_suffix='', inverter=None,
               task_batch_size=1, **kwargs):
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
    provenance['model'] = archive.name
    provenance['max_displacement'] = max_displacement
    provenance['crop'] = crop

    self.src = SrcDir(src_path, tgt_path, 
                      src_mask_path, tgt_mask_path, 
                      src_mask_mip, tgt_mask_mip, 
                      src_mask_val, tgt_mask_val)
    src_cv = self.src['src_img'][0]
    print("source_patch is", src_path)
    print("tgt_patch is", tgt_path)
    # info = DstDir.create_info_batch(src_cv, mip_range, max_displacement, 2,
    #                                 256, self.process_low_mip)
    info = DstDir.create_info(src_cv, mip_range, max_displacement)
    self.dst = {}
    self.tgt_radius = tgt_radius
    self.tgt_range = range(-tgt_radius, tgt_radius+1)
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
    self.inverter = inverter 
    self.pool = ThreadPool(threads)
    self.threads = threads
    self.task_batch_size = task_batch_size

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
                               inverse=False, relative=False, to_tensor=True,
                               field_cache = {}):
    """Compose a pairwise field at src_z with a previously composed field at tgt_z

    Args:
       src_z: int of source section index
       tgt_z: int of source section index
       F_cv: MiplessCloudVolume for the composed field
       bbox: BoundingBox for operation
       mip: int for MIP level
       inverse: bool indicating whether to left-compose the next field in the list
       relative: bool indicating whether returned field should be in relative space
       to_tensor: bool indicating whether return object should be a Tensor
       field_cache: dict storing previously composed fields; fields are stored as
         Tensors in relative space

    Returns:
       a field object, as specified by relative & to_tensor
    """
    z_offset = src_z - tgt_z
    f_cv = self.dst[z_offset].for_read('field')
    if inverse:
      f_z, F_z = src_z, src_z 
    else:
      f_z, F_z = src_z, tgt_z
    f = self.get_field(f_cv, f_z, bbox, mip, relative=True, to_tensor=to_tensor)
    tmp_key = self.create_key(bbox, F_z)
    if tmp_key in field_cache:
        print("{0} in FIELD_CACHE".format(tmp_key))
        F = field_cache[tmp_key]
    else:
        print("{0} NOT in FIELD_CACHE".format(tmp_key))
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
    """Retrieve vector field from CloudVolume.

    Args
      CV: MiplessCloudVolume storing vector field as MIP0 residuals in X,Y,Z,2 order
      Z: int for section index
      BBOX: BoundingBox for X & Y extent of the field to retrieve
      MIP: int for resolution at which to pull the vector field
      RELATIVE: bool indicating whether to convert MIP0 residuals to relative residuals
        from [-1,1] based on residual location within shape of the BBOX
      TO_TENSOR: bool indicating whether to return FIELD as a torch tensor

    Returns
      FIELD: vector field with dimensions of BBOX at MIP, with RELATIVE residuals &
        as TO_TENSOR, using convention (Z,Y,X,2) 

    Note that the grid convention for torch.grid_sample is (N,H,W,2), where the
    components in the final dimension are (x,y). We are NOT altering it here.
    """
    x_range = bbox.x_range(mip=mip)
    y_range = bbox.y_range(mip=mip)
    print('get_field from {bbox}, z={z}, MIP{mip} to {path}'.format(bbox=bbox,
                                 z=z, mip=mip, path=cv.path))
    field = cv[mip][x_range[0]:x_range[1], y_range[0]:y_range[1], z]
    field = np.transpose(field, (2,0,1,3))
    if relative:
      field = self.abs_to_rel_residual(field, bbox, mip)
    if to_tensor:
      field = torch.from_numpy(field)
      return field.to(device=self.device)
    else:
      return field 

  def save_vector_patch(self, cv, z, field, bbox, mip):
    """Save vector field to CloudVolume.

    Args
      CV: MiplessCloudVolume to store vector field as MIP0 residuals in X,Y,Z,2 order
      Z: int for section index
      FIELD: ndarray vector field with dimensions of BBOX at MIP with absolute MIP0 
        residuals, using grid_sample convention of (Z,Y,X,2), where the components in 
        the final dimension are (x,y).
      BBOX: BoundingBox for X & Y extent of the field to be stored
      MIP: int for resolution at which to store the vector field
    """
    # field = field.data.cpu().numpy() 
    x_range = bbox.x_range(mip=mip)
    y_range = bbox.y_range(mip=mip)
    field = np.transpose(field, (1,2,0,3))
    print('save_vector_patch at {bbox}, z={z}, MIP{mip} to {path}'.format(bbox=bbox,
                                 z=z, mip=mip, path=cv.path))
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

  def break_into_chunks_v2(self, bbox, chunk_size, offset, mip, render=False):
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
    processing_chunk = chunk_size
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

  def create_key(self, bbox, z):
    """Create string for cache key from BBOX & Z
    """
    return '{0}, {1}'.format(str(bbox.__str__(mip=0)), str(z))

  def vector_vote(self, z_range, read_F_cv, write_F_cv, bbox, mip, inverse, T=1,
                        negative_offsets=False, serial_operation=False):
    """Compute consensus vector field using pairwise vector fields with earlier sections. 

    Vector voting requires that vector fields be composed to a common section
    before comparison: inverse=False means that the comparison will be based on 
    composed vector fields F_{z,compose_start}, while inverse=True will be
    F_{compose_start,z}.

    Args:
       z_range: list of ints, indicating sections that will be sequentially processed
       tgt_range: list of ints, indicating the set of offsets of composed fields to use
        in the comparison
       read_F_cv: MiplessCloudVolume where the composed fields will be read
       write_F_cv: MiplessCloudVolume where the composed fields will be written
       bbox: BoundingBox, the region of interest over which to vote
       mip: int, the data MIP level
       inverse: bool, indicates the direction of composition to use 
       T: float for the temperature of the softmin used during comparison
       negative_offsets: bool indicating to use offsets less than 0 (z-i <-- z)
       serial_operation: bool indicating to if a previously composed field is 
        not necessary
    """
    print('vector_vote, negative_offset={0}, serial_operation={1}'.format(
                                                  negative_offsets, serial_operation))
    field_cache = {}
    oldest_z_offset = self.tgt_radius
    tgt_range = range(self.tgt_radius, 0, -1)
    if negative_offsets:
      oldest_z_offset = -self.tgt_radius
      tgt_range = range(-self.tgt_radius, 0)
    for z in z_range:
        fields = []
        for z_offset in tgt_range:
          src_z = z
          tgt_z = src_z - z_offset
          if inverse:
            src_z, tgt_z = tgt_z, src_z
          if serial_operation:
            f_cv = self.dst[z_offset].for_read('field')
            F = self.get_field(f_cv, src_z, bbox, mip, relative=False, to_tensor=True)
          else:
            F = self.get_composed_field(src_z, tgt_z, read_F_cv, bbox, mip,
                                        inverse=inverse, relative=False,
                                        to_tensor=True,
                                        field_cache=field_cache)
          fields.append(F)

        field = vector_vote(fields, T=T)
        delete_z = z - oldest_z_offset
        delete_key = self.create_key(bbox, delete_z)
        if delete_key in field_cache:
            del field_cache[delete_key]
            print("DELETE {0} from FIELD_CACHE".format(delete_key))
        tmp_key = self.create_key(bbox, z)
        print("PUT {0} in FIELD_CACHE".format(tmp_key))
        # Note: field_cache stores RELATIVE fields
        field_cache[tmp_key] = self.abs_to_rel_residual(field, bbox, mip)
        field = field.data.cpu().numpy() 
        self.save_vector_patch(write_F_cv, z, field, bbox, mip)

  # def vector_vote_single_section(self, z, read_F_cv, write_F_cv, bbox, mip, inverse, T=1):
  #   """Compute consensus vector field using pairwise vector fields with earlier sections. 

  #   Vector voting requires that vector fields be composed to a common section
  #   before comparison: inverse=False means that the comparison will be based on 
  #   composed vector fields F_{z,compose_start}, while inverse=True will be
  #   F_{compose_start,z}.

  #   Args:
  #      z: int, section whose pairwise vector fields will be used
  #      compose_start: int, the first pairwise vector field to use in calculating
  #        any composed vector fields
  #      bbox: BoundingBox, the region of interest over which to vote
  #      mip: int, the data MIP level
  #      inverse: bool, indicates the direction of composition to use 
  #      T: float for temperature of the softmin used for vector pair differences
  #   """
  #   fields = []
  #   for z_offset in range(self.tgt_radius, 0, -1):
  #     src_z = z
  #     tgt_z = src_z - z_offset
  #     if inverse:
  #       src_z, tgt_z = tgt_z, src_z
  #     if self.serial_operation:
  #       f_cv = self.dst[z_offset].for_read('field')
  #       F = self.get_field(f_cv, src_z, bbox, mip, relative=False, to_tensor=True)
  #     else:
  #       F = self.get_composed_field(src_z, tgt_z, read_F_cv, bbox, mip,
  #                                   inverse=inverse, relative=False,
  #                                   to_tensor=True)
  #     fields.append(F)

  #   field = vector_vote(fields, T=T)
  #   field = field.data.cpu().numpy() 
  #   self.save_vector_patch(write_F_cv, z, field, bbox, mip)

  def invert_field(self, z, src_cv, dst_cv, out_bbox, mip, optimizer=False):
    """Compute the inverse vector field for a given OUT_BBOX
    """
    crop = self.crop_amount
    precrop_bbox = deepcopy(out_bbox)
    precrop_bbox.uncrop(crop, mip=mip)
    f = self.get_field(src_cv, z, precrop_bbox, mip,
                           relative=True, to_tensor=True)
    print('invert_field shape: {0}'.format(f.shape))
    start = time()
    if optimizer: 
      invf = invert(f)[:,crop:-crop, crop:-crop,:]    
    else:
      invf = self.inverter(f)[:,crop:-crop, crop:-crop,:]
    print('invf shape: {0}'.format(invf.shape))
    end = time()
    print (": {} sec".format(end - start))
    # assert(torch.all(torch.isnan(invf)))
    invf = invf.data.cpu().numpy() 
    self.save_residual_patch(dst_cv, z, invf, out_bbox, mip) 

  def compute_residual_patch(self, src_z, src_cv, tgt_z, tgt_cv, field_cv, bbox, mip):
    """Predict vector field that will warp section at SOURCE_Z to section at TARGET_Z
    within OUT_PATCH_BBOX at MIP. Vector field will be stored at SOURCE_Z, using DST at
    SOURCE_Z - TARGET_Z. 

    Args
      src_z: int of section to be warped
      src_cv: MiplessCloudVolume with source image      
      tgt_z: int of section to be warped to
      tgt_cv: MiplessCloudVolume with target image
      field_cv: MiplessCloudVolume of where to write the output field
      bbox: BoundingBox for region of both sections to process
      mip: int of MIP level to use for OUT_PATCH_BBOX 
    """
    print ("Computing residual for region {0}, {1} <-- {2}.".format(bbox.__str__(mip=0),
                                                              tgt_z, src_z), flush=True)
    precrop_patch_bbox = deepcopy(bbox)
    precrop_patch_bbox.uncrop(self.crop_amount, mip=mip)

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
    field = field.data.cpu().numpy() 
    self.save_vector_patch(field_cv, src_z, field, bbox, mip)

    # if self.write_intermediaries and residuals is not None and cum_residuals is not None:
    #   mip_range = range(self.process_low_mip+self.size-1, self.process_low_mip-1, -1)
    #   for res_mip, res, cumres in zip(mip_range, residuals[1:], cum_residuals[1:]):
    #       crop = self.crop_amount // 2**(res_mip - self.process_low_mip)   
    #       self.save_residual_patch('res', src_z, src_z_offset, res, crop, bbox, res_mip)
    #       self.save_residual_patch('cumres', src_z, src_z_offset, cumres, crop, bbox, res_mip)
    #       if self.upsample_residuals:
    #         crop = self.crop_amount   
    #         res = self.scale_residuals(res, res_mip, self.process_low_mip)
    #         self.save_residual_patch('resup', src_z, z_offset, res, crop, bbox, 
    #                                  self.process_low_mip)
    #         cumres = self.scale_residuals(cumres, res_mip, self.process_low_mip)
    #         self.save_residual_patch('cumresup', src_z, z_offset, cumres, crop, 
    #                                  bbox, self.process_low_mip)

    #   print('encoding size: {0}'.format(len(encodings)))
    #   for k, enc in enumerate(encodings):
    #       mip = self.process_low_mip + k
    #       # print('encoding shape @ idx={0}, mip={1}: {2}'.format(k, mip, enc.shape))
    #       crop = self.crop_amount // 2**k
    #       enc = enc[:,:,crop:-crop, crop:-crop].permute(2,3,0,1)
    #       enc = enc.data.cpu().numpy()
    #       
    #       def write_encodings(j_slice, z):
    #         x_range = bbox.x_range(mip=mip)
    #         y_range = bbox.y_range(mip=mip)
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
    """Convert vector field from relative space [-1,1] to absolute MIP0 space
    """
    return field * (field.shape[-2] / 2) * (2**mip)

  def abs_to_rel_residual(self, abs_residual, patch, mip):
    """Convert vector field from absolute MIP0 space to relative space [-1,1]
    """
    x_fraction = patch.x_size(mip=0) * 0.5
    y_fraction = patch.y_size(mip=0) * 0.5
    rel_residual = deepcopy(abs_residual)
    rel_residual[:, :, :, 0] /= x_fraction
    rel_residual[:, :, :, 1] /= y_fraction
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

  def profile_field(self, field):
      min_x = math.floor(np.min(field[...,0]))
      min_y = math.floor(np.min(field[...,1]))
      return np.float32([min_x, min_y])

  def adjust_bbox(self, bbox, dis):
      influence_bbox = deepcopy(bbox)
      x_range = influence_bbox.x_range(mip=0)
      y_range = influence_bbox.y_range(mip=0)
      #print("x_range is", x_range, "y_range is", y_range)
      new_bbox = BoundingBox(x_range[0] - dis[0], x_range[1] - dis[0],
                                   y_range[0] - dis[1], y_range[1] - dis[1],
                                   mip=0)
      #print(new_bbox.x_range(mip=0), new_bbox.y_range(mip=0))
      return new_bbox

  def gridsample_cv(self, image_cv, field_cv, bbox, z, mip):
      f =  self.get_field(field_cv, z, bbox, mip, relative=False,
                          to_tensor=True) 
      x_range = bbox.x_range(mip=0)
      y_range = bbox.y_range(mip=0)
      if torch.min(field) == 0 and torch.max(field) == 0:
          image = self.get_image(image_cv, z, bbox, mip,
                                 adjust_contrast=False, to_tensor=True)
          return image
      else:
          #im_off = 10240
          #f += im_off
          distance = self.profile_field(f)
          distance = (distance//(2**mip)) * 2**mip
          #print("x_range is", x_range, "y_range is", y_range)
          #new_bbox = BoundingBox(x_range[0] - im_off, x_range[1] - im_off,
          #                       y_range[0] - im_off, y_range[1] - im_off, mip=0)
          #new_bbox = self.adjust_bbox(new_bbox, distance)
          new_bbox = self.adjust_bbox(bbox, distance)
          print("distance is", distance)
          f = f - distance.to(device = self.device)
          #f = f - distance
          res = self.abs_to_rel_residual(f, bbox, mip)
          #res = torch.from_numpy(res)
          field = res.to(device = self.device)
          #print("field shape is", field.shape)
          image = self.get_image(image_cv, z, new_bbox, mip,
                                 adjust_contrast=False, to_tensor=True)
          #print("image shape is", image.shape)
          if 'src_mask' in self.src:
              mask_cv = self.src['src_mask']
              mask = self.get_mask(mask_cv, z, new_bbox,
                                   src_mip=self.src.src_mask_mip,
                                   dst_mip=mip, valid_val=self.src.src_mask_val)
              image = image.masked_fill_(mask, 0)
          image = gridsample_residual(image, field, padding_mode='zeros')
          return image

  def warp_using_gridsample_cv(self, src_z, field_cv, field_z, bbox, mip):
      influence_bbox = deepcopy(bbox)
      influence_bbox.uncrop(self.max_displacement, mip=0)
      mip_disp = int(self.max_displacement / 2**mip)
      src_cv = self.src['src_img']
      image = self.gridsample_cv(src_cv, field_cv, influence_bbox, field_z, mip)
      if self.disable_cuda:
        image = image.numpy()[:,:,mip_disp:-mip_disp,mip_disp:-mip_disp]
      else:
        image = image.cpu().numpy()[:,:,mip_disp:-mip_disp,mip_disp:-mip_disp]
      return image

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
      print("warped")
    else:
      print ("not warping")
    # print('warp_image image1.shape: {0}'.format(image.shape))
    if self.disable_cuda:
      image = image.numpy()[:,:,mip_disp:-mip_disp,mip_disp:-mip_disp]
    else:
      image = image.cpu().numpy()[:,:,mip_disp:-mip_disp,mip_disp:-mip_disp]
    # print('warp_image image3.shape: {0}'.format(image.shape))
    return image

  def warp_patch_batch(self, src_z, field_cv, field_z, bbox, mip, batch):
    """Non-chunk warping

    From BBOX at MIP, warp image at SRC_Z in CloudVolume SRC_CV using
    field at FIELD_Z in CloudVolume FIELD_CV.
    """
    influence_bbox = deepcopy(bbox)
    influence_bbox.uncrop(self.max_displacement, mip=0)
    start = time()
    image_batch = []
    print("z range in warp_patch_batch", src_z, src_z + batch)
    for z in range(src_z, src_z + batch):
        field = self.get_field(field_cv, z, influence_bbox, mip,
                               relative=True, to_tensor=True)
        mip_disp = int(self.max_displacement / 2**mip)
        src_cv = self.src['src_img']
        image = self.get_image(src_cv, z, influence_bbox, mip,
                               adjust_contrast=False, to_tensor=True)
        if 'src_mask' in self.src:
          mask_cv = self.src['src_mask']
          mask = self.get_mask(mask_cv, z, influence_bbox, 
                               src_mip=self.src.src_mask_mip,
                               dst_mip=mip, valid_val=self.src.src_mask_val)
          image = image.masked_fill_(mask, 0)
        # no need to warp if flow is identity since warp introduces noise
        if torch.min(field) != 0 or torch.max(field) != 0:
          image = gridsample_residual(image, field, padding_mode='zeros')
          print("warped")
        else:
          print ("not warping")
        if self.disable_cuda:
          image = image.numpy()[0, 0, mip_disp:-mip_disp, mip_disp:-mip_disp]
        else:
          image = image.cpu().numpy()[0, 0, mip_disp:-mip_disp, mip_disp:-mip_disp]
        print('warp_image image.shape: {0}'.format(image.shape))
        image_batch.append(image)
    return np.array(image_batch)

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
    # mip_field = self.rel_to_abs_residual(mip_field, image_mip)
    if self.disable_cuda:
      image = image.numpy()[:,:,mip_disp:-mip_disp,mip_disp:-mip_disp]
      # f = mip_field.numpy()[:,mip_disp:-mip_disp,mip_disp:-mip_disp,:]
    else:
      image = image.cpu().numpy()[:,:,mip_disp:-mip_disp,mip_disp:-mip_disp]
      # f = mip_field.cpu().numpy()[:,mip_disp:-mip_disp,mip_disp:-mip_disp,:]
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
    patch = np.transpose(float_patch, (2,3,0,1))
    #print("----------------z is", z, "save image patch at mip", mip, "range", x_range, y_range, "range at mip0", bbox.x_range(mip=0), bbox.y_range(mip=0))
    if to_uint8:
      patch = (np.multiply(patch, 255)).astype(np.uint8)
    cv[mip][x_range[0]:x_range[1], y_range[0]:y_range[1], z] = patch

  def save_image_patch_batch(self, cv, z_range, float_patch, bbox, mip, to_uint8=True):
    x_range = bbox.x_range(mip=mip)
    y_range = bbox.y_range(mip=mip)
    print("type of float_patch", type(float_patch), "shape", float_patch.shape)
    patch = np.transpose(float_patch, (2,1,0))[..., np.newaxis]
    if to_uint8:
        patch = (np.multiply(patch, 255)).astype(np.uint8)
    print("patch shape", patch.shape)
    cv[mip][x_range[0]:x_range[1], y_range[0]:y_range[1],
            z_range[0]:z_range[1]] = patch

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
    #data = None
    #while data is None:
    #  try:
    #    data_ = cv[src_mip][x_range[0]:x_range[1], y_range[0]:y_range[1], z]
    #    data = data_
    #  except AttributeError as e:
    #    pass 
    #
    data = np.transpose(data, (2,3,0,1))
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
    if self.distributed and len(chunks) > self.task_batch_size * 4:
        tasks = []
        for i in range(0, len(chunks), self.task_batch_size * 4):
            task_patches = []
            for j in range(i, min(len(chunks), i + self.task_batch_size * 4)):
                task_patches.append(chunks[j])
            tasks.append(make_copy_task_message(z, dst_cv, dst_z, task_patches, mip=mip))
        self.pool.map(self.task_handler.send_message, tasks)
        self.task_handler.wait_until_ready()
    else: 
        #for patch_bbox in chunks:
        def chunkwise(patch_bbox):
          src_cv = self.src['src_img']
          if 'src_mask' in self.src:
            mask_cv = self.src['src_mask']
            raw_patch = self.get_image(src_cv, z, patch_bbox, mip,
                                        adjust_contrast=False, to_tensor=True)
            raw_mask = self.get_mask(mask_cv, z, patch_bbox, 
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

  def render(self, src_z, field_cv, field_z, dst_cv, dst_z, bbox, mip, wait=True):
    """Chunkwise render

    Warp the image in BBOX at MIP and SRC_Z in CloudVolume dir at SRC_Z_OFFSET, 
    using the field at FIELD_Z in CloudVolume dir at FIELD_Z_OFFSET, and write 
    the result to DST_Z in CloudVolume dir at DST_Z_OFFSET. Chunk BBOX 
    appropriately.
    """
    self.total_bbox = bbox
    print('Rendering src_z={0} @ MIP{1} to dst_z={2}'.format(src_z, mip, dst_z), flush=True)
    start = time()
    chunks = self.break_into_chunks(bbox, self.dst[0].dst_chunk_sizes[mip],
                                    self.dst[0].dst_voxel_offsets[mip], mip=mip, render=True)
    if self.distributed:
        tasks = []
        for i in range(0, len(chunks), self.task_batch_size):
            task_patches = []
            for j in range(i, min(len(chunks), i + self.task_batch_size)):
                task_patches.append(chunks[j])
            tasks.append(make_render_task_message(src_z, field_cv, field_z, task_patches, 
                                                   mip, dst_cv, dst_z))
        self.pool.map(self.task_handler.send_message, tasks)
        if wait:
          self.task_handler.wait_until_ready()
    else:
        def chunkwise(patch_bbox):
          warped_patch = self.warp_patch(src_z, field_cv, field_z, patch_bbox, mip)
          # print('warp_image render.shape: {0}'.format(warped_patch.shape))
          self.save_image_patch(dst_cv, dst_z, warped_patch, patch_bbox, mip)
        self.pool.map(chunkwise, chunks)
    end = time()
    print (": {} sec".format(end - start))

  def render_batch(self, src_z, field_cv, field_z, dst_cv, dst_z, bbox, mip,
                   batch):
    """Chunkwise render

    Warp the image in BBOX at MIP and SRC_Z in CloudVolume dir at SRC_Z_OFFSET, 
    using the field at FIELD_Z in CloudVolume dir at FIELD_Z_OFFSET, and write 
    the result to DST_Z in CloudVolume dir at DST_Z_OFFSET. Chunk BBOX 
    appropriately.
    """
    self.total_bbox = bbox
    print('Rendering src_z={0} @ MIP{1} to dst_z={2}'.format(src_z, mip, dst_z), flush=True)
    start = time()
    print("chunk_size: ", self.dst[0].dst_chunk_sizes[mip], self.dst[0].dst_voxel_offsets[mip])
    chunks = self.break_into_chunks_v2(bbox, self.dst[0].dst_chunk_sizes[mip],
                                    self.dst[0].dst_voxel_offsets[mip], mip=mip, render=True)
    if self.distributed:
        tasks = []
        for i in range(0, len(chunks), self.task_batch_size):
            task_patches = []
            for j in range(i, min(len(chunks), i + self.task_batch_size)):
                task_patches.append(chunks[j])
            tasks.append(make_batch_render_message(src_z, field_cv, field_z, task_patches,
                                                   mip, dst_cv, dst_z, batch))
        self.pool.map(self.task_handler.send_message, tasks)
        self.task_handler.wait_until_ready()
    else:
        def chunkwise(patch_bbox):
          warped_patch = self.warp_patch_batch(src_z, field_cv, field_z,
                                               patch_bbox, mip, batch)
          self.save_image_patch_batch(dst_cv, (dst_z, dst_z + batch), warped_patch, patch_bbox, mip)
        self.pool.map(chunkwise, chunks)
    end = time()
    print (": {} sec".format(end - start))

  def render_grid_cv(self, src_z, field_cv, field_z, dst_cv, dst_z, bbox, mip):
    """Chunkwise render

    Warp the image in BBOX using CloudVolume grid_sample
    """
    self.total_bbox = bbox
    print('Rendering src_z={0} @ MIP{1} to dst_z={2}'.format(src_z, mip, dst_z), flush=True)
    start = time()
    chunks = self.break_into_chunks(bbox, self.dst[0].dst_chunk_sizes[mip],
                                    self.dst[0].dst_voxel_offsets[mip], mip=mip, render=True)
    #prof_chunk = chunks[len(chunks)//2]
    #f =  self.get_field(field_cv, src_z, prof_chunk, mip, relative=False,
    #                    to_tensor=False)
    ##f += 10240
    #distance = self.profile_field(f)
    #distance = (distance//(2**mip)) * 2**mip
    if self.distributed:
        tasks = []
        for i in range(0, len(chunks), self.task_batch_size):
            task_patches = []
            for j in range(i, min(len(chunks), i + self.task_batch_size)):
                task_patches.append(chunks[j])
            tasks.append(make_render_cv_task_message(src_z, field_cv, field_z, task_patches,
                                                      mip, dst_cv, dst_z))
        self.pool.map(self.task_handler.send_message, tasks)
        self.task_handler.wait_until_ready()
    else:
        def chunkwise(patch_bbox):
          warped_patch = self.warp_using_gridsample_cv(src_z, field_cv,
                                                       field_z, patch_bbox,
                                                       mip)
          # print('warp_image render.shape: {0}'.format(warped_patch.shape))
          self.save_image_patch(dst_cv, dst_z, warped_patch, patch_bbox, mip)
        self.pool.map(chunkwise, chunks)
    end = time()
    print (": {} sec".format(end - start))

#  def render(self, src_z, field_cv, field_z, dst_cv, dst_z, bbox, mip):
#    """Chunkwise render
#
#    Warp the image in BBOX at MIP and SRC_Z in CloudVolume dir at SRC_Z_OFFSET, 
#    using the field at FIELD_Z in CloudVolume dir at FIELD_Z_OFFSET, and write 
#    the result to DST_Z in CloudVolume dir at DST_Z_OFFSET. Chunk BBOX 
#    appropriately.
#    """
#    print('Rendering src_z={0} @ MIP{1} to dst_z={2}'.format(src_z, mip, dst_z), flush=True)
#    start = time()
#    chunks = self.break_into_chunks(bbox, self.dst[0].dst_chunk_sizes[mip],
#                                    self.dst[0].dst_voxel_offsets[mip], mip=mip, render=True)
#    if self.distributed:
#        for patch in chunks:
#            render_task = make_render_task_message(src_z, field_cv, field_z, patch, 
#                                                   mip, dst_cv, dst_z)
#            self.task_handler.send_message(render_task)
#        self.task_handler.wait_until_ready()
#    else:
#        def chunkwise(patch_bbox):
#          warped_patch = self.warp_patch(src_z, field_cv, field_z, patch_bbox, mip)
#          # print('warp_image render.shape: {0}'.format(warped_patch.shape))
#          self.save_image_patch(dst_cv, dst_z, warped_patch, patch_bbox, mip)
#        self.pool.map(chunkwise, chunks)
#    end = time()
#    print (": {} sec".format(end - start))

  def low_mip_render(self, src_z, field_cv, field_z, dst_cv, dst_z, bbox, image_mip, vector_mip):
    start = time()
    chunks = self.break_into_chunks(bbox, self.dst[0].dst_chunk_sizes[image_mip],
                                    self.dst[0].dst_voxel_offsets[image_mip], mip=image_mip, render=True)
    print("low_mip_render at MIP{0} ({1} chunks)".format(image_mip,len(chunks)))
    if self.distributed:
        tasks = []
        for i in range(0, len(chunks), self.task_batch_size):
            task_patches = []
            for j in range(i, min(len(chunks), i + self.task_batch_size)):
                task_patches.append(chunks[j])
            tasks.append(make_render_low_mip_task_message(src_z, field_cv, field_z, 
                                                           task_patches, image_mip, 
                                                           vector_mip, dst_cv, dst_z))
        self.pool.map(self.task_handler.send_message, tasks)
        self.task_handler.wait_until_ready()
    else:
        def chunkwise(patch_bbox):
          warped_patch = self.warp_patch_at_low_mip(src_z, field_cv, field_z, patch_bbox, image_mip, vector_mip)
          print('warp_image render.shape: {0}'.format(warped_patch.shape))
          self.save_image_patch(dst_cv, dst_z, warped_patch, patch_bbox, image_mip)
          # self.save_vector_patch(out_field_cv, dst_z, up_field, patch_bbox, image_mip)
        self.pool.map(chunkwise, chunks)
    end = time()
    print (": {} sec".format(end - start))
  
  def downsample_range(self, cv, z_range, bbox, source_mip, target_mip):
    """Downsample a range of sections, downsampling a given MIP across all sections
       before proceeding to the next higher MIP level.
    
    Args:
       cv: MiplessCloudVolume where images will be loaded and written
       z_range: list of ints for section indices that will be downsampled
       bbox: BoundingBox for region to be downsampled in each section
       source_mip: int for MIP level of the data to be initially loaded
       target_mip: int for MIP level after which downsampling will stop
    """
    for mip in range(source_mip, target_mip):
      print('downsample_range from {src} to {tgt}'.format(src=source_mip, tgt=target_mip))
      for z in z_range:
        self.downsample(cv, z, bbox, mip, mip+1, wait=False)
      if self.distributed:
        self.task_handler.wait_until_ready()
    
  def downsample(self, cv, z, bbox, source_mip, target_mip, wait=True):
    """Chunkwise downsample

    For the CloudVolume dirs at Z_OFFSET, warp the SRC_IMG using the FIELD for
    section Z in region BBOX at MIP. Chunk BBOX appropriately and save the result
    to DST_IMG.
    """
    print ("Downsampling {} from mip {} to mip {}".format(bbox.__str__(mip=0), source_mip, target_mip))
    for m in range(source_mip+1, target_mip+1):
      chunks = self.break_into_chunks(bbox, self.dst[0].dst_chunk_sizes[m],
                                      self.dst[0].dst_voxel_offsets[m], mip=m, render=True)
      if self.distributed and len(chunks) > self.task_batch_size * 4:
          tasks = []
          print("Distributed downsampling to mip", m, len(chunks)," chunks")
          for i in range(0, len(chunks), self.task_batch_size * 4):
              task_patches = []
              for j in range(i, min(len(chunks), i + self.task_batch_size * 4)):
                  task_patches.append(chunks[j])
              tasks.append(make_downsample_task_message(cv, z, task_patches, mip=m))
          self.pool.map(self.task_handler.send_message, tasks)
          if wait:
            self.task_handler.wait_until_ready()
      else:
          def chunkwise(patch_bbox):
            print ("Downsampling {} to mip {}".format(patch_bbox.__str__(mip=0), m))
            downsampled_patch = self.downsample_patch(cv, z, patch_bbox, m-1)
            self.save_image_patch(cv, z, downsampled_patch, patch_bbox, m)
          self.pool.map(chunkwise, chunks)

#  def downsample(self, cv, z, bbox, source_mip, target_mip):
#    """Chunkwise downsample
#
#    For the CloudVolume dirs at Z_OFFSET, warp the SRC_IMG using the FIELD for
#    section Z in region BBOX at MIP. Chunk BBOX appropriately and save the result
#    to DST_IMG.
#    """
#    print ("Downsampling {} from mip {} to mip {}".format(bbox.__str__(mip=0), source_mip, target_mip))
#    for m in range(source_mip+1, target_mip + 1):
#      chunks = self.break_into_chunks(bbox, self.dst[0].dst_chunk_sizes[m],
#                                      self.dst[0].dst_voxel_offsets[m], mip=m, render=True)
#      if self.distributed and len(chunks) > self.task_batch_size * 4:
#          print("Distributed downsampling to mip", m, len(chunks)," chunks")
#          #for c in chunks:
#          #  print ("distributed Downsampling {} to mip {}".format(c.__str__(mip=0), m))
#          for c in chunks:
#              downsample_task = make_downsample_task_message(cv, z, c, mip=m)
#              self.task_handler.send_message(downsample_task) 
#          self.task_handler.wait_until_ready()
#      else:
#          #for c in chunks:
#          #  print ("Downsampling {} to mip {}".format(c.__str__(mip=0), m))
#          def chunkwise(patch_bbox):
#            print ("Downsampling {} to mip {}".format(patch_bbox.__str__(mip=0), m))
#            downsampled_patch = self.downsample_patch(cv, z, patch_bbox, m-1)
#            self.save_image_patch(cv, z, downsampled_patch, patch_bbox, m)
#          self.pool.map(chunkwise, chunks)


  def render_section_all_mips(self, src_z, field_cv, field_z, dst_cv, dst_z, bbox, mip, wait=True):
    self.render(src_z, field_cv, field_z, dst_cv, dst_z, bbox, self.render_low_mip, wait=wait)
    # self.render_grid_cv(src_z, field_cv, field_z, dst_cv, dst_z, bbox, self.render_low_mip)
    self.downsample(dst_cv, dst_z, bbox, self.render_low_mip, self.render_high_mip, wait=wait)
  
  def render_to_low_mip(self, src_z, field_cv, field_z, dst_cv, dst_z, bbox, image_mip, vector_mip):
      self.low_mip_render(src_z, field_cv, field_z, dst_cv, dst_z, bbox, image_mip, vector_mip)
      self.downsample(dst_cv, dst_z, bbox, image_mip, self.render_high_mip)

  def compute_section_pair_residuals(self, src_z, src_cv, tgt_z, tgt_cv, field_cv,
                                           bbox, mip):
    """Chunkwise vector field inference for section pair

    Args:
       src_z: int for section index of source image
       src_cv: MiplessCloudVolume where source image to be loaded
       tgt_z: int for section index of target image
       tgt_cv: MiplessCloudVolume where target image to be loaded
       field_cv: MiplessCloudVolume where output vector field will be written
       bbox: BoundingBox for region where source and target image will be loaded,
        and where the resulting vector field will be written
       mip: int for MIP level images will be loaded and field will be stored at
    """
    start = time()
    chunks = self.break_into_chunks(bbox, self.dst[0].vec_chunk_sizes[mip],
                                    self.dst[0].vec_voxel_offsets[mip], mip=mip)
    print ("compute residuals between {} to slice {} at mip {} ({} chunks)".
           format(src_z, tgt_z, mip, len(chunks)), flush=True)
    if self.distributed:
      tasks = []
      for patch_bbox in chunks:
        tasks.append(make_residual_task_message(src_z, src_cv, tgt_z, tgt_cv, 
                                                   field_cv, patch_bbox, mip))
      self.pool.map(self.task_handler.send_message, tasks)
    else:
      def chunkwise(patch_bbox):
      #FIXME Torch runs out of memory
      #FIXME batchify download and upload
        self.compute_residual_patch(src_z, src_cv, tgt_z, tgt_cv, 
                                    field_cv, patch_bbox, mip)
      self.pool.map(chunkwise, chunks)
    end = time()
    print (": {} sec".format(end - start))
    
  def count_box(self, bbox, mip):    
    chunks = self.break_into_chunks(bbox, self.dst[0].dst_chunk_sizes[mip],
                                      self.dst[0].dst_voxel_offsets[mip], mip=mip, render=True)
    total_chunks = len(chunks)
    self.image_pixels_sum =np.zeros(total_chunks)
    self.field_sf_sum =np.zeros((total_chunks, 2), dtype=np.float32)

  def invert_field_chunkwise(self, z, src_cv, dst_cv, bbox, mip, optimizer=False):
    """Chunked-processing of vector field inversion 
    
    Args:
       z: section of fields to weight
       src_cv: CloudVolume for forward field
       dst_cv: CloudVolume for inverted field
       bbox: boundingbox of region to process
       mip: field MIP level
       optimizer: bool to use the Optimizer instead of the net
    """
    start = time()
    chunks = self.break_into_chunks(bbox, self.dst[0].vec_chunk_sizes[mip],
                                    self.dst[0].vec_voxel_offsets[mip], mip=mip)
    print("Vector field inversion for slice {0} @ MIP{1} ({2} chunks)".
           format(z, mip, len(chunks)), flush=True)
    if self.distributed:
        tasks = []
        for patch_bbox in chunks:
          tasks.append(make_invert_field_task_message(z, src_cv, dst_cv, patch_bbox, 
                                                      mip, optimizer))
        self.pool.map(self.task_handler.send_message, tasks)
    else: 
    #for patch_bbox in chunks:
        def chunkwise(patch_bbox):
          self.invert_field(z, src_cv, dst_cv, patch_bbox, mip)
        self.pool.map(chunkwise, chunks)
    end = time()
    print (": {} sec".format(end - start))

  # def vector_vote_single_section_chunkwise(self, z, read_F_cv, write_F_cv, bbox, mip, inverse, T=-1):
  #   """Chunked-processing of vector voting

  #   Args:
  #      z: section of fields to weight
  #      read_F_cv: CloudVolume with the vectors to compose against
  #      write_F_cv: CloudVolume where the resulting vectors will be written 
  #      bbox: boundingbox of region to process
  #      mip: field MIP level
  #      T: softmin temperature (default will be 2**mip)
  #   """
  #   start = time()
  #   chunks = self.break_into_chunks(bbox, self.dst[0].vec_chunk_sizes[mip],
  #                                   self.dst[0].vec_voxel_offsets[mip], mip=mip)
  #   print("Vector voting for slice {0} @ MIP{1} {2} ({3} chunks)".
  #          format(z, mip, 'INVERSE' if inverse else 'FORWARD', len(chunks)), flush=True)

  #   if self.distributed:
  #       for patch_bbox in chunks:
  #           vector_vote_task = make_vector_vote_task_message(z, read_F_cv, write_F_cv,
  #                                                            patch_bbox, mip, inverse, T) 
  #           self.task_handler.send_message(vector_vote_task)
  #       self.task_handler.wait_until_ready()
  #   #for patch_bbox in chunks:
  #   else:
  #       def chunkwise(patch_bbox):
  #           self.vector_vote([z], read_F_cv, write_F_cv, patch_bbox, mip, inverse=inverse, T=T)
  #       self.pool.map(chunkwise, chunks)
  #   end = time()
  #   print (": {} sec".format(end - start))

  def vector_vote_chunkwise(self, z_range, read_F_cv, write_F_cv, bbox, mip, inverse, 
                                  T=-1, negative_offsets=False, serial_operation=False):
    """Chunked-processing of vector voting

    Args:
       z: list of ints for sections to be vector voted
       read_F_cv: CloudVolume with the vectors to compose against
       write_F_cv: CloudVolume where the resulting vectors will be written 
       bbox: boundingbox of region to process
       mip: field MIP level
       T: softmin temperature (default will be 2**mip)
       negative_offsets: bool indicating whether to use offsets less than 0 (z-i <-- z)
       serial_operation: bool indicating to if a previously composed field is 
        not necessary
    """
    start = time()
    chunks = self.break_into_chunks(bbox, self.dst[0].vec_chunk_sizes[mip],
                                    self.dst[0].vec_voxel_offsets[mip], mip=mip)
    print("Vector voting for slices {0} @ MIP{1} {2} ({3} chunks)".
           format(z_range, mip, 'INVERSE' if inverse else 'FORWARD', len(chunks)), flush=True)

    if self.distributed:
        tasks = []
        for patch_bbox in chunks:
            tasks.append(make_vector_vote_task_message(z_range, read_F_cv, write_F_cv,
                                                             patch_bbox, mip, inverse, T, 
                                                             negative_offsets, 
                                                             serial_operation))
        self.pool.map(self.task_handler.send_message, tasks)
        self.task_handler.wait_until_ready()
    #for patch_bbox in chunks:
    else:
        def chunkwise(patch_bbox):
            self.vector_vote(z_range, read_F_cv, write_F_cv, patch_bbox, mip, 
                             inverse=inverse, T=T, negative_offsets=negative_offsets,
                             serial_operation=serial_operation)
        self.pool.map(chunkwise, chunks)
    end = time()
    print (": {} sec".format(end - start))

  def multi_match(self, z, forward_match, reverse_match, render=True):
    """Match z to all sections within tgt_radius

    Args:
        z: int for section index
        forward_match: bool indicating whether to match z to z-i
        reverse_match: bool indicating whether to match z to z+i
        render: bool indicating whether to render section
    """
    bbox = self.total_bbox
    mip = self.process_low_mip
    tgt_range = []
    src_cv = self.src['src_img']
    tgt_cv = self.src['tgt_img']
    if forward_match:
      tgt_range.extend(range(self.tgt_range[-1], 0, -1)) 
    if reverse_match:
      tgt_range.extend(range(self.tgt_range[0], 0, 1)) 
    for z_offset in tgt_range:
      if z_offset != 0:
        src_z = z
        tgt_z = src_z - z_offset
        field_cv = self.dst[z_offset].for_write('field')
        self.compute_section_pair_residuals(src_z, src_cv, tgt_z, tgt_cv, field_cv, 
                                            bbox, mip) 
        if render:
          field_cv = self.dst[z_offset].for_read('field')
          dst_cv = self.dst[z_offset].for_write('dst_img')
          self.render_section_all_mips(src_z, field_cv, src_z, dst_cv, tgt_z, bbox, mip)

  def generate_pairwise(self, z_range, bbox, forward_match, reverse_match, 
                              render_match=False, batch_size=1):
    """Create all pairwise matches for each SRC_Z in Z_RANGE to each TGT_Z in TGT_RADIUS
  
    Args:
        z_range: list of z indices to be matches 
        bbox: BoundingBox object for bounds of 2D region
        forward_match: bool indicating whether to match from z to z-i
          for i in range(tgt_radius)
        reverse_match: bool indicating whether to match from z to z+i
          for i in range(tgt_radius)
        render_match: bool indicating whether to separately render out
          each aligned section before compiling vector fields with voting
          (useful for debugging)
        batch_size: (for distributed only) int describing how many sections to issue 
          multi-match tasks for, before waiting for all tasks to complete
    """
    self.total_bbox = bbox
    mip = self.process_low_mip
    batch_count = 0
    start = 0
    for z in z_range:
      start = time()
      batch_count += 1 
      self.multi_match(z, forward_match=forward_match, reverse_match=reverse_match, 
                       render=render_match)
      if batch_count == batch_size and self.distributed:
        print('generate_pairwise waiting for {batch} section(s)'.format(batch=batch_size))
        self.task_handler.wait_until_ready()
        end = time()
        print (": {} sec".format(end - start))
        batch_count = 0
    # report on remaining sections after batch 
    if batch_count > 0 and self.distributed: 
      print('generate_pairwise waiting for {batch} section(s)'.format(batch=batch_size))
      self.task_handler.wait_until_ready()
      end = time()
      print (": {} sec".format(end - start))
    #if self.p_render:
    #    self.task_handler.wait_until_ready()
 
  def compose_pairwise(self, z_range, compose_start, bbox, mip,
                             forward_compose=True, inverse_compose=True, 
                             negative_offsets=False, serial_operation=False):
    """Combine pairwise vector fields in TGT_RADIUS using vector voting, while composing
    with earliest section at COMPOSE_START.

    Args
       z_range: list of ints (assumed to be monotonic & sequential)
       compose_start: int of earliest section used in composition
       bbox: BoundingBox defining chunk region
       mip: int for MIP level of data
       forward_compose: bool, indicating whether to compose with forward transforms
       inverse_compose: bool, indicating whether to compose with inverse transforms
       negative_offsets: bool indicating whether to use offsets less than 0 (z-i <-- z)
       serial_operation: bool indicating to if a previously composed field is 
        not necessary
    """
    self.total_bbox = bbox
    T = 2**mip
    print('softmin temp: {0}'.format(T))
    if forward_compose:
      self.dst[0].add_composed_cv(compose_start, inverse=False)
    if inverse_compose: 
      self.dst[0].add_composed_cv(compose_start, inverse=True)
    # for z in z_range:
    #     write_F_k = self.dst[0].get_composed_key(compose_start, inverse=False)
    #     write_invF_k = self.dst[0].get_composed_key(compose_start, inverse=True)
    #     read_F_k = write_F_k
    #     read_invF_k = write_invF_k
    #      
    #     if forward_compose:
    #       read_F_cv = self.dst[0].for_read(read_F_k)
    #       write_F_cv = self.dst[0].for_write(write_F_k)
    #       self.vector_vote_chunkwise(z, read_F_cv, write_F_cv, bbox, mip, inverse=False, T=T)
    #     if inverse_compose:
    #       read_F_cv = self.dst[0].for_read(read_invF_k)
    #       write_F_cv = self.dst[0].for_write(write_invF_k)
    #       self.vector_vote_chunkwise(z, read_F_cv, write_F_cv, bbox, mip, inverse=True, T=T)
    write_F_k = self.dst[0].get_composed_key(compose_start, inverse=False)
    write_invF_k = self.dst[0].get_composed_key(compose_start, inverse=True)
    read_F_k = write_F_k
    read_invF_k = write_invF_k
     
    if forward_compose:
      read_F_cv = self.dst[0].for_read(read_F_k)
      write_F_cv = self.dst[0].for_write(write_F_k)
      self.vector_vote_chunkwise(z_range, read_F_cv, write_F_cv, bbox, mip, 
                                 inverse=False, T=T, negative_offsets=negative_offsets,
                                 serial_operation=serial_operation)
    if inverse_compose:
      read_F_cv = self.dst[0].for_read(read_invF_k)
      write_F_cv = self.dst[0].for_write(write_invF_k)
      self.vector_vote_chunkwise(z_range, read_F_cv, write_F_cv, bbox, mip, 
                                 inverse=False, T=T, negative_offsets=negative_offsets,
                                 serial_operation=serial_operation)

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
      regF = regF.data.cpu().numpy() 
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
        tasks = []
        for patch_bbox in chunks:
            tasks.append(make_regularize_task_message(z_range[0], z_range[-1],
                                                      dir_z, patch_bbox,
                                                      mip, sigma))
        self.pool.map(self.task_handler.send_message, tasks)
        self.task_handler.wait_until_ready()
    else:
        #for patch_bbox in chunks:
        def chunkwise(patch_bbox):
          self.regularize_z(z_range, dir_z, patch_bbox, mip, sigma=sigma)
        self.pool.map(chunkwise, chunks)
    end = time()
    print (": {} sec".format(end - start))

  def handle_residual_task(self, message):
    src_z = message['src_z']
    src_cv = DCV(message['src_cv']) 
    tgt_z = message['tgt_z']
    tgt_cv = DCV(message['tgt_cv']) 
    field_cv = DCV(message['field_cv']) 
    patch_bbox = deserialize_bbox(message['patch_bbox'])
    mip = message['mip']
    self.compute_residual_patch(src_z, src_cv, tgt_z, tgt_cv, field_cv, patch_bbox, mip)

  def handle_render_task_cv(self, message):
    src_z = message['z']
    patches  = [deserialize_bbox(p) for p in message['patches']]
    #patches  = deserialize_bbox(message['patches'])
    field_cv = DCV(message['field_cv']) 
    mip = message['mip']
    field_z = message['field_z']
    dst_cv = DCV(message['dst_cv'])
    dst_z = message['dst_z']
    def chunkwise(patch_bbox):
      print ("Rendering {} at mip {}".format(patch_bbox.__str__(mip=0), mip),
              end='', flush=True)
      warped_patch = self.warp_using_gridsample_cv(src_z, field_cv, field_z, patch_bbox, mip)
      self.save_image_patch(dst_cv, dst_z, warped_patch, patch_bbox, mip)
    self.pool.map(chunkwise, patches)

  def handle_batch_render_task(self, message):
    src_z = message['z']
    patches  = [deserialize_bbox(p) for p in message['patches']]
    batch = message['batch']
    field_cv = DCV(message['field_cv'])
    mip = message['mip']
    field_z = message['field_z']
    dst_cv = DCV(message['dst_cv'])
    dst_z = message['dst_z']
    def chunkwise(patch_bbox):
      print ("Rendering {} at mip {}".format(patch_bbox.__str__(mip=0), mip),
              end='', flush=True)
      warped_patch = self.warp_patch_batch(src_z, field_cv, field_z,
                                           patch_bbox, mip, batch)
      self.save_image_patch_batch(dst_cv, (dst_z, dst_z + batch),
                                  warped_patch, patch_bbox, mip)
    self.pool.map(chunkwise, patches)

  def handle_render_task(self, message):
    src_z = message['z']
    patches  = [deserialize_bbox(p) for p in message['patches']]
    #patches  = deserialize_bbox(message['patches'])
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
    #warped_patch = self.warp_patch(src_z, field_cv, field_z, patches, mip)
    #self.save_image_patch(dst_cv, dst_z, warped_patch, patches, mip)

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
        raw_mask = self.get_mask(mask_cv, z, patch_bbox, 
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
    #patches  = deserialize_bbox(message['patches'])
    patches  = [deserialize_bbox(p) for p in message['patches']]
    mip = message['mip']
    #downsampled_patch = self.downsample_patch(cv, z, patches, mip - 1)
    #self.save_image_patch(cv, z, downsampled_patch, patches, mip)
    def chunkwise(patch_bbox):
      downsampled_patch = self.downsample_patch(cv, z, patch_bbox, mip - 1)
      self.save_image_patch(cv, z, downsampled_patch, patch_bbox, mip)
    self.pool.map(chunkwise, patches)

  # def handle_vector_vote(self, message):
  #     z = message['z']
  #     #z_end = message['z_end']
  #     read_F_cv = DCV(message['read_F_cv'])
  #     write_F_cv =DCV(message['write_F_cv'])
  #     #chunks = [deserialize_bbox(p) for p in message['patch_bbox']]
  #     chunks = deserialize_bbox(message['patch_bbox'])
  #     mip = message['mip']
  #     inverse = message['inverse']
  #     T = message['T']
  #     #z_range = range(z, z_end)
  #     self.vector_vote(z, read_F_cv, write_F_cv, chunks, mip, inverse=inverse, T=T)

  def handle_vector_vote(self, message):
      z_start = message['z_start']
      z_end = message['z_end']
      read_F_cv = DCV(message['read_F_cv'])
      write_F_cv =DCV(message['write_F_cv'])
      #chunks = [deserialize_bbox(p) for p in message['patch_bbox']]
      chunks = deserialize_bbox(message['patch_bbox'])
      mip = message['mip']
      inverse = message['inverse']
      T = message['T']
      negative_offsets = message['negative_offsets']
      serial_operation = message['serial_operation']
      z_range = range(z_start, z_end+1)
      self.vector_vote(z_range, read_F_cv, write_F_cv, chunks, mip, inverse=inverse, 
                       T=T, negative_offsets=negative_offsets, 
                       serial_operation=serial_operation)

  def handle_regularize(self, message):
      z_start = message['z_start']
      z_end = message['z_end']
      compose_start = message['compose_start']
      patch_bbox = deserialize_bbox(message['patch_bbox'])
      mip = message['mip']
      sigma = message['sigma']
      z_range = range(z_start, z_end+1)
      self.regularize_z(z_range, compose_start, patch_bbox, mip, sigma=sigma)

  def handle_invert(self, message):
      z = message['z']
      src_cv = DCV(message['src_cv'])
      dst_cv = DCV(message['dst_cv'])
      patch_bbox = deserialize_bbox(message['patch_bbox'])
      mip = message['mip']
      optimizer = message['optimizer']
      self.invert_field(z, src_cv, dst_cv, patch_bbox, mip, optimizer)

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
    elif task_type == 'render_task_cv':
      self.handle_render_task_cv(body)
    elif task_type == 'batch_render_task':
      self.handle_batch_render_task(body)
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
    # elif task_type == 'batch_vvote_task':
    #   self.handle_batch_vvote(body)
    elif task_type == 'regularize_task':
      self.handle_regularize(body)
    elif task_type == 'invert_task':
      self.handle_invert(body)
    else:
      raise Exception("Unsupported task type '{}' received from queue '{}'".format(task_type,
                                                                 self.task_handler.queue_name))

  def listen_for_tasks(self):
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
