import concurrent.futures
from copy import deepcopy, copy
from functools import partial
import json
import math
import os
from os.path import join
from time import time, sleep

from pathos.multiprocessing import ProcessPool, ThreadPool
from threading import Lock

from cloudvolume import Storage
from cloudvolume.lib import Vec
import numpy as np
import scipy
import scipy.ndimage
from skimage.morphology import disk as skdisk
from skimage.filters.rank import maximum as skmaximum 
from taskqueue import TaskQueue, LocalTaskQueue
import torch
from torch.nn.functional import interpolate
import torch.nn as nn

from normalizer import Normalizer
from vector_vote import vector_vote, get_diffs, weight_diffs, \
                        compile_field_weights, weighted_sum_fields
from temporal_regularization import create_field_bump
from cpc import cpc 
from utilities.helpers import save_chunk, crop, upsample, gridsample_residual, \
                              np_downsample, invert, compose_fields, upsample_field, \
                              is_identity
from boundingbox import BoundingBox, deserialize_bbox

from pathos.multiprocessing import ProcessPool, ThreadPool
from threading import Lock
from pathlib import Path
from utilities.archive import ModelArchive

import torch.nn as nn
from taskqueue import TaskQueue
import tasks

class Aligner:
  def __init__(self, threads=1, queue_name=None, task_batch_size=1, **kwargs):
    print('Creating Aligner object')

    self.distributed = (queue_name != None)
    self.queue_name = queue_name
    self.task_queue = None
    if queue_name:
      self.task_queue = TaskQueue(queue_name=queue_name, n_threads=0)
    
    self.chunk_size = (1024, 1024)
    self.device = torch.device('cuda')

    self.model_archives = {}
    
    # self.pool = None #ThreadPool(threads)
    self.threads = threads
    self.task_batch_size = task_batch_size

  ##########################
  # Chunking & BoundingBox #
  ##########################

  def break_into_chunks(self, bbox, chunk_size, offset, mip, max_mip=12):
    """Break bbox into list of chunks with chunk_size, given offset for all data 

    Args:
       bbox: BoundingBox for region to be broken into chunks
       chunk_size: tuple for dimensions of chunk that bbox will be broken into;
         will be set to min(chunk_size, self.chunk_size)
       offset: tuple for x,y origin for the entire dataset, from which chunks
         will be aligned
       mip: int for MIP level at which bbox is defined
       max_mip: int for the maximum MIP level at which the bbox is valid
    """
    if chunk_size[0] > self.chunk_size[0] or chunk_size[1] > self.chunk_size[1]:
      chunk_size = self.chunk_size 

    raw_x_range = bbox.x_range(mip=mip)
    raw_y_range = bbox.y_range(mip=mip)
    
    x_chunk = chunk_size[0]
    y_chunk = chunk_size[1]
    
    x_offset = offset[0]
    y_offset = offset[1]

    x_remainder = ((raw_x_range[0] - x_offset) % x_chunk)
    y_remainder = ((raw_y_range[0] - y_offset) % y_chunk)

    calign_x_range = [raw_x_range[0] - x_remainder, raw_x_range[1]]
    calign_y_range = [raw_y_range[0] - y_remainder, raw_y_range[1]]

    chunks = []
    for xs in range(calign_x_range[0], calign_x_range[1], chunk_size[0]):
      for ys in range(calign_y_range[0], calign_y_range[1], chunk_size[1]):
        chunks.append(BoundingBox(xs, xs + chunk_size[0],
                                 ys, ys + chunk_size[1],
                                 mip=mip, max_mip=max_mip))
    return chunks

  def adjust_bbox(self, bbox, dis):
      padded_bbox = deepcopy(bbox)
      x_range = padded_bbox.x_range(mip=0)
      y_range = padded_bbox.y_range(mip=0)
      #print("x_range is", x_range, "y_range is", y_range)
      new_bbox = BoundingBox(x_range[0] + dis[1], x_range[1] + dis[1],
                                   y_range[0] + dis[0], y_range[1] + dis[0],
                                   mip=0)
      #print(new_bbox.x_range(mip=0), new_bbox.y_range(mip=0))
      return new_bbox

  ##############
  # IO methods #
  ##############

  def get_model_archive(self, model_path):
    """Load a model stored in the repo with its relative path

    TODO: evict old models from self.models

    Args:
       model_path: str for relative path to model directory

    Returns:
       the ModelArchive at that model_path
    """
    if model_path in self.model_archives:
      print('Loading model {0} from cache'.format(model_path))
      return self.model_archives[model_path]
    else:
      print('Adding model {0} to the cache'.format(model_path))
      path = Path(model_path)
      model_name = path.stem
      archive = ModelArchive(model_name)
      self.model_archives[model_path] = archive
      return archive

  #######################
  # Image IO + handlers #
  #######################

  def get_mask(self, cv, z, bbox, src_mip, dst_mip, valid_val, to_tensor=True):
    data = self.get_data(cv, z, bbox, src_mip=src_mip, dst_mip=dst_mip, 
                             to_float=False, to_tensor=to_tensor, normalizer=None)
    return data == valid_val

  def get_image(self, cv, z, bbox, mip, to_tensor=True, normalizer=None):
    print('get_image for {0}'.format(bbox.stringify(z)))
    return self.get_data(cv, z, bbox, src_mip=mip, dst_mip=mip, to_float=True, 
                             to_tensor=to_tensor, normalizer=normalizer)

  def get_masked_image(self, image_cv, z, bbox, image_mip, mask_cv, mask_mip, mask_val,
                             to_tensor=True, normalizer=None):
    """Get image with mask applied
    """
    image = self.get_image(image_cv, z, bbox, image_mip,
                           to_tensor=to_tensor, normalizer=normalizer)
    if mask_cv is not None:
      print('masking image')
      mask = self.get_mask(mask_cv, z, bbox, 
                           src_mip=mask_mip,
                           dst_mip=image_mip, valid_val=mask_val)
      image = image.masked_fill_(mask, 0)
    return image

  def get_composite_image(self, cv, z_list, bbox, mip, to_tensor=True): 
    """Collapse 3D image into a 2D image, replacing black pixels in the first 
        z slice with the nearest nonzero pixel in other slices.
    
    Args:
       cv: MiplessCloudVolume where images are stored
       z_list: list of ints that will be processed in order 
       bbox: BoundingBox defining data range
       mip: int MIP level of the data to process
       adjust_contrast: output will be normalized
       to_tensor: output will be torch.tensor
       #TODO normalizer: callable function to adjust the contrast of the image
    """
    z_start = np.min(z_range)
    z_stop = np.max(z_range)+1
    z_range = range(z_start, z_stop) 
    img = self.get_data_range(cv, z_range, bbox, src_mip=mip, dst_mip=mip)
    z = z_list[0]
    o = img[z-z_start, ...]
    for z in z_list[1:]:
      o[o <= 1] = img[z-z_start, ...][o <= 1]
    return o

  def get_data(self, cv, z, bbox, src_mip, dst_mip, to_float=True, 
                     to_tensor=True, normalizer=None):
    """Retrieve CloudVolume data. Returns 4D ndarray or tensor, BxCxWxH
    
    Args:
       cv_key: string to lookup CloudVolume
       bbox: BoundingBox defining data range
       src_mip: mip of the CloudVolume data
       dst_mip: mip of the output mask (dictates whether to up/downsample)
       to_float: output should be float32
       to_tensor: output will be torch.tensor
       normalizer: callable function to adjust the contrast of the image

    Returns:
       image from CloudVolume in region bbox at dst_mip, with contrast adjusted,
       if normalizer is specified, and as a uint8 or float32 torch tensor or numpy, 
       as specified
    """
    x_range = bbox.x_range(mip=src_mip)
    y_range = bbox.y_range(mip=src_mip)
    data = cv[src_mip][x_range[0]:x_range[1], y_range[0]:y_range[1], z]
    data = np.transpose(data, (2,3,0,1))
    if to_float:
      data = np.divide(data, float(255.0), dtype=np.float32)
    if normalizer is not None:
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
  
  def get_data_range(self, cv, z_range, bbox, src_mip, dst_mip, to_tensor=True):
    """Retrieve CloudVolume data. Returns 4D tensor, BxCxWxH
    
    Args:
       cv_key: string to lookup CloudVolume
       bbox: BoundingBox defining data range
       src_mip: mip of the CloudVolume data
       dst_mip: mip of the output mask (dictates whether to up/downsample)
       to_tensor: output will be torch.tensor
       #TODO normalizer: callable function to adjust the contrast of the image
    """
    x_range = bbox.x_range(mip=src_mip)
    y_range = bbox.y_range(mip=src_mip)
    data = cv[src_mip][x_range[0]:x_range[1], y_range[0]:y_range[1], z_range]
    data = np.transpose(data, (2,3,0,1))
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

  def save_image(self, float_patch, cv, z, bbox, mip, to_uint8=True):
    x_range = bbox.x_range(mip=mip)
    y_range = bbox.y_range(mip=mip)
    patch = np.transpose(float_patch, (2,3,0,1))
    #print("----------------z is", z, "save image patch at mip", mip, "range", x_range, y_range, "range at mip0", bbox.x_range(mip=0), bbox.y_range(mip=0))
    if to_uint8:
      patch = (np.multiply(patch, 255)).astype(np.uint8)
    cv[mip][x_range[0]:x_range[1], y_range[0]:y_range[1], z] = patch

  def save_image_batch(self, cv, z_range, float_patch, bbox, mip, to_uint8=True):
    x_range = bbox.x_range(mip=mip)
    y_range = bbox.y_range(mip=mip)
    print("type of float_patch", type(float_patch), "shape", float_patch.shape)
    patch = np.transpose(float_patch, (2,3,0,1))
    # patch = np.transpose(float_patch, (2,1,0))[..., np.newaxis]
    if to_uint8:
        patch = (np.multiply(patch, 255)).astype(np.uint8)
    print("patch shape", patch.shape)
    cv[mip][x_range[0]:x_range[1], y_range[0]:y_range[1],
            z_range[0]:z_range[1]] = patch

  #######################
  # Field IO + handlers #
  #######################

  def get_field(self, cv, z, bbox, mip, relative=False, to_tensor=True, as_int16=True):
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
    if as_int16:
      field = np.float32(field) / 4
    if relative:
      field = self.abs_to_rel_residual(field, bbox, mip)
    if to_tensor:
      field = torch.from_numpy(field)
      return field.to(device=self.device)
    else:
      return field 

  def get_composed_field(self, f_cv, g_cv, f_z, g_z, bbox, f_mip, g_mip, dst_mip,
                               pad=2048):
    """Compose chunk of two field cloudvolumes, such that f(g(x)) at dst_mip

    Args:
       f_z: int section index of the f CloudVolume
       g_z: int section index of the g CloudVolume
       f_cv: MiplessCloudVolume of left-hand vector field
       g_cv: MiplessCloudVolume of right-hand vector field
       bbox: BoundingBox for region to process
       f_mip: MIP of left-hand vector field
       g_mip: MIP of right-hand vector field
       dst_mip: MIP of the output vector field, such that min(f_mip, g_mip) >= dst_mip
       pad: int for amount of MIP0 padding to use before processing

    Returns:
       the composed vector field at MIP min(f_mip, g_mip) in absolute space for size
       of bbox
    """
    padded_bbox = deepcopy(bbox)
    padded_bbox.uncrop(pad, mip=0)
    crop = pad // 2**dst_mip
    f = self.get_field(f_cv, f_z, padded_bbox, f_mip, relative=True, to_tensor=True)
    g = self.get_field(g_cv, g_z, padded_bbox, g_mip, relative=True, to_tensor=True)
    if dst_mip < g_mip:
      g = upsample_field(g, g_mip, dst_mip)
    elif dst_mip < f_mip:
      f = upsample_field(f, f_mip, dst_mip)
    h = compose_fields(f, g)
    h = self.rel_to_abs_residual(h, dst_mip)
    h = h[:,crop:-crop,crop:-crop,:]
    return h

  def save_field(self, field, cv, z, bbox, mip, relative, as_int16=True):
    """Save vector field to CloudVolume.

    Args
      field: ndarray vector field with dimensions of bbox at mip with absolute MIP0 
        residuals, using grid_sample convention of (Z,Y,X,2), where the components in 
        the final dimension are (x,y).
      cv: MiplessCloudVolume to store vector field as MIP0 residuals in X,Y,Z,2 order
      z: int for section index
      bbox: BoundingBox for X & Y extent of the field to be stored
      mip: int for resolution at which to store the vector field
      relative: bool indicating whether to convert MIP0 residuals to relative residuals
        from [-1,1] based on residual location within shape of the bbox 
      as_int16: bool indicating whether vectors should be saved as int16
    """
    if relative: 
      field = field * (field.shape[-2] / 2) * (2**mip)
    # field = field.data.cpu().numpy() 
    x_range = bbox.x_range(mip=mip)
    y_range = bbox.y_range(mip=mip)
    field = np.transpose(field, (1,2,0,3))
    print('save_field for {0} at MIP{1} to {2}'.format(bbox.stringify(z),
                                                       mip, cv.path))
    if as_int16:
      if(np.max(field) > 8192 or np.min(field) < -8191):
        print('Value in field is out of range of int16 max: {}, min: {}'.format(
                                               np.max(field),np.min(field)), flush=True)
      field = np.int16(field * 4)
    #print("**********field shape is ", field.shape, type(field[0,0,0,0]))
    cv[mip][x_range[0]:x_range[1], y_range[0]:y_range[1], z] = field

  def rel_to_abs_residual(self, field, mip):    
    """Convert vector field from relative space [-1,1] to absolute MIP0 space
    """
    return field * (field.shape[-2] / 2) * (2**mip)

  def abs_to_rel_residual(self, field, bbox, mip):
    """Convert vector field from absolute MIP0 space to relative space [-1,1]
    """
    x_fraction = bbox.x_size(mip=0) * 0.5
    y_fraction = bbox.y_size(mip=0) * 0.5
    rel_residual = deepcopy(field)
    rel_residual[:, :, :, 0] /= x_fraction
    rel_residual[:, :, :, 1] /= y_fraction
    return rel_residual

  def avg_field(self, field):
    favg = field.sum() / torch.nonzero(field).size(0)
    return favg

  def profile_field(self, field):
    avg_x = self.avg_field(field[0,...,0])
    avg_y = self.avg_field(field[0,...,1])
    return torch.from_numpy(np.float32([avg_x, avg_y]))

  #############################
  # CloudVolume chunk methods #
  #############################

  def compute_field_chunk(self, model_path, src_cv, tgt_cv, src_z, tgt_z, bbox, mip, pad, 
                          src_mask_cv=None, src_mask_mip=0, src_mask_val=0,
                          tgt_mask_cv=None, tgt_mask_mip=0, tgt_mask_val=0):
    """Run inference with SEAMLeSS model on two images stored as CloudVolume regions.

    Args:
      model_path: str for relative path to model directory
      src_z: int of section to be warped
      src_cv: MiplessCloudVolume with source image      
      tgt_z: int of section to be warped to
      tgt_cv: MiplessCloudVolume with target image
      bbox: BoundingBox for region of both sections to process
      mip: int of MIP level to use for bbox 
      pad: int for amount of padding to add to the bbox before processing
      mask_cv: MiplessCloudVolume with mask to be used for both src & tgt image
      
    Returns:
      field with MIP0 residuals with the shape of bbox at MIP mip
    """
    archive = self.get_model_archive(model_path)
    model = archive.model
    normalizer = archive.preprocessor
    print('compute_field for {0} to {1}'.format(bbox.stringify(src_z),
                                                bbox.stringify(tgt_z)))
    padded_bbox = deepcopy(bbox)
    padded_bbox.uncrop(pad, mip=mip)

    src_patch = self.get_masked_image(src_cv, src_z, padded_bbox, mip,
                                mask_cv=src_mask_cv, mask_mip=src_mask_mip,
                                mask_val=src_mask_val,
                                to_tensor=True, normalizer=normalizer)
    tgt_patch = self.get_masked_image(tgt_cv, tgt_z, padded_bbox, mip,
                                mask_cv=tgt_mask_cv, mask_mip=tgt_mask_mip,
                                mask_val=tgt_mask_val,
                                to_tensor=True, normalizer=normalizer)

    # model produces field in relative coordinates
    field = model(src_patch, tgt_patch)
    field = self.rel_to_abs_residual(field, mip)
    field = field[:,pad:-pad,pad:-pad,:]
    return field

  def vector_vote_chunk(self, pairwise_cvs, vvote_cv, z, bbox, mip, 
                        inverse=False, softmin_temp=-1, serial=True):
    """Compute consensus vector field using pairwise vector fields with earlier sections. 

    Vector voting requires that vector fields be composed to a common section
    before comparison: inverse=False means that the comparison will be based on 
    composed vector fields F_{z,compose_start}, while inverse=True will be
    F_{compose_start,z}.

    TODO:
       Reimplement field_cache

    Args:
       pairwise_cvs: dict of MiplessCloudVolumes, indexed by their z_offset
       vvote_cv: MiplessCloudVolume where vector-voted field will be stored 
       z: int for section index to be vector voted 
       bbox: BoundingBox for region where all fields will be loaded/written
       mip: int for MIP level of fields
       softmin_temp: softmin temperature (default will be 2**mip)
       inverse: bool indicating if pairwise fields are to be treated as inverse fields 
       serial: bool indicating to if a previously composed field is 
        not necessary
    """
    fields = []
    for z_offset, f_cv in pairwise_cvs.items():
      if serial:
        F = self.get_field(f_cv, z, bbox, mip, relative=False, to_tensor=True)
      else:
        G_cv = vvote_cv
        if inverse:
          f_z = z+z_offset
          G_z = z+z_offset
          F = self.get_composed_field(f_cv, G_cv, f_z, G_z, bbox, mip, mip, mip)
        else:
          f_z = z
          G_z = z+z_offset
          F = self.get_composed_field(G_cv, f_cv, G_z, f_z, bbox, mip, mip, mip)
      fields.append(F)
    return vector_vote(fields, T=softmin_temp)

  def invert_field(self, z, src_cv, dst_cv, bbox, mip, pad, model_path):
    """Compute the inverse vector field for a given bbox 

    Args:
       z: int for section index to be processed
       src_cv: MiplessCloudVolume where the field to be inverted is stored
       dst_cv: MiplessCloudVolume where the inverted field will be stored
       bbox: BoundingBox for region to be processed
       mip: int for MIP level to be processed
       pad: int for additional bbox padding to use during processing
       model_path: string for relative path to the inverter model; if blank, then use
        the runtime optimizer
    """
    padded_bbox = deepcopy(bbox)
    padded_bbox.uncrop(pad, mip=mip)
    f = self.get_field(src_cv, z, padded_bbox, mip,
                       relative=True, to_tensor=True, as_int16=as_int16)
    print('invert_field shape: {0}'.format(f.shape))
    start = time()
    if model_path:
      archive = self.get_model_archive(model_path)
      model = archive.model
      invf = model(f)
    else:
      # use optimizer if no model provided
      invf = invert(f)
    invf = self.rel_to_abs_residual(invf, mip=mip)
    invf = invf[:,pad:-pad, pad:-pad,:]    
    end = time()
    print (": {} sec".format(end - start))
    invf = invf.data.cpu().numpy() 
    self.save_field(dst_cv, z, invf, bbox, mip, relative=True, as_int16=as_int16) 

  def cloudsample_image(self, image_cv, field_cv, image_z, field_z, 
                              bbox, image_mip, field_mip, 
                              mask_cv=None, mask_mip=0, mask_val=0,
                              as_int16=True):
      """Wrapper for torch.nn.functional.gridsample for CloudVolume image objects

      Args:
         z: int for section index to warp
         image_cv: MiplessCloudVolume storing the image
         field_cv: MiplessCloudVolume storing the vector field
         bbox: BoundingBox for output region to be warped
         image_mip: int for MIP of the image
         field_mip: int for MIP of the vector field; must be >= image_mip.
          If field_mip > image_mip, the field will be upsampled.

      Returns:
         warped image with shape of bbox at MIP image_mip
      """
      assert(field_mip >= image_mip)
      pad = 2**(image_mip+1)
      padded_bbox = deepcopy(bbox)
      padded_bbox.uncrop(pad, mip=image_mip)
      f =  self.get_field(field_cv, field_z, padded_bbox, field_mip, relative=False,
                          to_tensor=True)
      x_range = bbox.x_range(mip=0)
      y_range = bbox.y_range(mip=0)
      if is_identity(f):
        image = self.get_image(image_cv, image_z, bbox, image_mip,
                               to_tensor=True, normalizer=None)
        if mask_cv is not None:
          mask = self.get_mask(mask_cv, image_z, bbox, 
                               src_mip=mask_mip,
                               dst_mip=image_mip, valid_val=mask_val)
          image = image.masked_fill_(mask, 0)
        return image
      else:
        distance = self.profile_field(f)
        distance = (distance//(2**field_mip)) * 2**field_mip
        new_bbox = self.adjust_bbox(padded_bbox, distance)
        f = f - distance.to(device = self.device)
        res = self.abs_to_rel_residual(f, padded_bbox, field_mip)
        field = res.to(device = self.device)
        if field_mip > image_mip:
          field = upsample_field(field, field_mip, image_mip)
        image = self.get_masked_image(image_cv, image_z, new_bbox, image_mip,
                                mask_cv=mask_cv, mask_mip=mask_mip,
                                mask_val=mask_val,
                                to_tensor=True, normalizer=None)
        image = gridsample_residual(image, field, padding_mode='zeros')
        image = image[:,:,pad:-pad,pad:-pad]
        return image

  # def cloudsample_compose(self, f_cv, g_cv, f_z, g_z, bbox, f_mip, g_mip, dst_mip,
  #                               pad=2048):
  #     """Wrapper for torch.nn.functional.gridsample for CloudVolume field objects.

  #     Gridsampling a field is a composition, such that f(g(x)).

  #     ** THIS METHOD HAS NOT BEEN TESTED **

  #     Args:
  #        f_cv: MiplessCloudVolume storing the vector field to do the warping 
  #        g_cv: MiplessCloudVolume storing the vector field to be warped
  #        bbox: BoundingBox for output region to be warped
  #        z: int for section index to warp
  #        f_mip: int for MIP of the warping field 
  #        g_mip: int for MIP of the field to be warped
  #        dst_mip: int for MIP of the desired output field

  #     Returns:
  #        composed field
  #     """
  #     padded_bbox = deepcopy(bbox)
  #     padded_bbox.uncrop(pad, mip=0)
  #     crop = pad // 2**dst_mip
  #     f = self.get_field(f_cv, f_z, padded_bbox, f_mip, relative=False,
  #                         to_tensor=True)
  #     x_range = bbox.x_range(mip=0)
  #     y_range = bbox.y_range(mip=0)
  #     if is_identity(f):
  #       h = self.get_field(g_cv, g_z, bbox, g_mip, relative=False,
  #                           to_tensor=True)
  #       return h 
  #     else:
  #       distance = self.profile_field(f)
  #       distance = (distance//(2**f_mip)) * 2**f_mip
  #       new_bbox = self.adjust_bbox(padded_bbox, distance)
  #       g = self.get_field(g_cv, g_z, new_bbox, g_mip, relative=True,
  #                           to_tensor=True)

  #       f = f - distance.to(device = self.device)
  #       f = self.abs_to_rel_residual(f, padded_bbox, f_mip)
  #       f = f.to(device = self.device)

  #       if dst_mip < g_mip:
  #         g = upsample_field(g, g_mip, dst_mip)
  #       if dst_mip < f_mip:
  #         f = upsample_field(f, f_mip, dst_mip)

  #       h = compose_fields(f, g)
  #       h = self.rel_to_abs_residual(h, dst_mip)
  #       h = h[:,crop:-crop,crop:-crop,:]
  #       return h

  def cloudsample_image_batch(self, z_range, image_cv, field_cv, 
                              bbox, image_mip, field_mip,
                              mask_cv=None, mask_mip=0, mask_val=0,
                              as_int16=True):
    """Warp a batch of sections using the cloudsampler 
   
    Args:
       z_range: list of ints for section indices to process
       image_cv: MiplessCloudVolume of source image
       field_cv: MiplesscloudVolume of vector field
       bbox: BoundingBox of output region
       image_mip: int for MIP of the source image
       field_mip: int for MIP of the vector field

    Returns:
       torch tensor of all images, concatenated along axis=0 
    """
    start = time()
    batch = []
    print("cloudsample_image_batch for z_range={0}".format(z_range))
    for z in z_range: 
      image = self.cloudsample_image(z, z, image_cv, field_cv, bbox, 
                                  image_mip, field_mip, 
                                  mask_cv=mask_cv, mask_mip=mask_mip, mask_val=mask_val, 
                                  as_int16=as_int16)
      batch.append(image)
    return torch.cat(batch, axis=0)

  def downsample(self, cv, z, bbox, mip):
    data = self.get_image(cv, z, bbox, mip, adjust_contrast=False, to_tensor=True)
    data = interpolate(data, scale_factor=0.5, mode='bilinear')
    return data.cpu().numpy()

  def cpc(self, src_z, tgt_z, src_cv, tgt_cv, bbox, src_mip, dst_mip):
    """Calculate the chunked pearson r between two chunks

    Args:
       src_z: int z index of one section to compare
       tgt_z: int z index of other section to compare
       src_cv: MiplessCloudVolume of source image
       tgt_cv: MiplessCloudVolume of target image
       bbox: BoundingBox of region to process
       src_mip: int MIP level of input src & tgt images
       dst_mip: int MIP level of output image, will dictate the size of the chunks
        used for the pearson r

    Returns:
       img for bbox at dst_mip containing pearson r at each pixel for the chunks
       in src & tgt images at src_mip
    """
    print('Compute CPC for {4} at MIP{0} to MIP{1}, {2}<-({2},{3})'.format(src_mip, 
                                                                   dst_mip, src_z, 
                                                                   tgt_z, 
                                                                   bbox.__str__(mip=0)))
    scale_factor = 2**(dst_mip - src_mip)
    src = self.get_image(src_cv, src_z, bbox, src_mip, adjust_contrast=False, 
                         to_tensor=True)
    tgt = self.get_image(tgt_cv, tgt_z, bbox, src_mip, adjust_contrast=False, 
                         to_tensor=True)
    return cpc(src, tgt, scale_factor, device=self.device)

  ######################
  # Dataset operations #
  ######################
  def copy(self, cm, src_cv, dst_cv, src_z, dst_z, bbox, mip, is_field=False,
                     mask_cv=None, mask_mip=0, mask_val=0, prefix=''):
    """Copy one CloudVolume to another

    Args:
       cm: CloudManager that corresponds to the src_cv, tgt_cv, and field_cv
       model_path: str for relative path to ModelArchive
       src_z: int for section index of source image
       dst_z: int for section index of destination image
       src_cv: MiplessCloudVolume where source image is stored 
       dst_cv: MiplessCloudVolume where destination image will be stored 
       bbox: BoundingBox for region where source and target image will be loaded,
        and where the resulting vector field will be written
       mip: int for MIP level images will be loaded and field will be stored at
       field: bool indicating whether this is a field CloudVolume to copy
       mask_cv: MiplessCloudVolume where source mask is stored
       mask_mip: int for MIP level at which source mask is stored
       mask_val: int for pixel value in the mask that should be zero-filled
       prefix: str used to write "finished" files for each task 
        (only used for distributed)

    Returns:
       a list of CopyTasks
    """
    chunks = self.break_into_chunks(bbox, cm.dst_chunk_sizes[mip],
                                    cm.dst_voxel_offsets[mip], mip=mip, 
                                    max_mip=cm.num_scales)
    if prefix == '':
      prefix = '{}_{}'.format(mip, dst_z)
    batch = []
    for chunk in chunks: 
      batch.append(tasks.CopyTask(src_cv, dst_cv, src_z, dst_z, chunk, mip, 
                                  is_field, mask_cv, mask_mip, mask_val, prefix))
    return batch

  def compute_field(self, cm, model_path, src_cv, tgt_cv, field_cv, 
                          src_z, tgt_z, bbox, mip, pad=2048, prefix=''):
    """Compute field to warp src section to tgt section 
  
    Args:
       cm: CloudManager that corresponds to the src_cv, tgt_cv, and field_cv
       model_path: str for relative path to ModelArchive
       src_cv: MiplessCloudVolume where source image to be loaded
       tgt_cv: MiplessCloudVolume where target image to be loaded
       field_cv: MiplessCloudVolume where output vector field will be written
       src_z: int for section index of source image
       tgt_z: int for section index of target image
       bbox: BoundingBox for region where source and target image will be loaded,
        and where the resulting vector field will be written
       mip: int for MIP level images will be loaded and field will be stored at
       pad: int for amount of padding to add to bbox before processing
       wait: bool indicating whether to wait for all tasks must finish before proceeding
       prefix: str used to write "finished" files for each task 
        (only used for distributed)
    """
    start = time()
    chunks = self.break_into_chunks(bbox, cm.dst_chunk_sizes[mip],
                                    cm.dst_voxel_offsets[mip], mip=mip, 
                                    max_mip=cm.num_scales)
    if prefix == '':
      prefix = '{}_{}_{}'.format(mip, src_z, tgt_z)
    batch = []
    for chunk in chunks:
      batch.append(tasks.ComputeFieldTask(model_path, src_cv, tgt_cv,
                                                   field_cv, src_z, tgt_z, chunk, 
                                                   mip, pad, prefix)) 
    return batch
  
  def render(self, cm, src_cv, field_cv, dst_cv, src_z, field_z, dst_z, 
                   bbox, src_mip, field_mip, mask_cv=None, mask_mip=0, 
                   mask_val=0, prefix=''):
    """Warp image in src_cv by field in field_cv and save result to dst_cv

    Args:
       cm: CloudManager that corresponds to the src_cv, field_cv, & dst_cv
       src_cv: MiplessCloudVolume where source image is stored 
       field_cv: MiplessCloudVolume where vector field is stored 
       dst_cv: MiplessCloudVolume where destination image will be written 
       src_z: int for section index of source image
       field_z: int for section index of vector field 
       dst_z: int for section index of destination image
       bbox: BoundingBox for region where source and target image will be loaded,
        and where the resulting vector field will be written
       src_mip: int for MIP level of src images 
       field_mip: int for MIP level of vector field; field_mip >= src_mip
       mask_cv: MiplessCloudVolume where source mask is stored
       mask_mip: int for MIP level at which source mask is stored
       mask_val: int for pixel value in the mask that should be zero-filled
       wait: bool indicating whether to wait for all tasks must finish before proceeding
       prefix: str used to write "finished" files for each task 
        (only used for distributed)
    """
    start = time()
    chunks = self.break_into_chunks(bbox, cm.dst_chunk_sizes[src_mip],
                                    cm.dst_voxel_offsets[src_mip], mip=src_mip, 
                                    max_mip=cm.num_scales)
    if prefix == '':
      prefix = '{}_{}_{}'.format(src_mip, src_z, dst_z)
    batch = []
    for chunk in chunks:
      batch.append(tasks.RenderTask(src_cv, field_cv, dst_cv, src_z, 
                       field_z, dst_z, chunk, src_mip, field_mip, mask_cv, 
                       mask_mip, mask_val, prefix))
    return batch

  def vector_vote(self, cm, pairwise_cvs, vvote_cv, z, bbox, mip, 
                        inverse=False, softmin_temp=-1, serial=True, 
                        prefix=''):
    """Compute consensus field from a set of vector fields

    Note: 
       tgt_z = src_z + z_offset

    Args:
       cm: CloudManager that corresponds to the src_cv, field_cv, & dst_cv
       pairwise_cvs: dict of MiplessCloudVolumes, indexed by their z_offset
       vvote_cv: MiplessCloudVolume where vector-voted field will be stored 
       z: int for section index to be vector voted 
       bbox: BoundingBox for region where all fields will be loaded/written
       mip: int for MIP level of fields
       softmin_temp: softmin temperature (default will be 2**mip)
       inverse: bool indicating if pairwise fields are to be treated as inverse fields 
       serial: bool indicating to if a previously composed field is 
        not necessary
       wait: bool indicating whether to wait for all tasks must finish before proceeding
       prefix: str used to write "finished" files for each task 
        (only used for distributed)
    """
    start = time()
    chunks = self.break_into_chunks(bbox, cm.vec_chunk_sizes[mip],
                                    cm.vec_voxel_offsets[mip], mip=mip)
    if prefix == '':
      prefix = '{}_{}'.format(mip, z)
    batch = []
    for chunk in chunks:
      batch.append(tasks.VectorVoteTask(deepcopy(pairwise_cvs), vvote_cv, z,
                                        chunk, mip, inverse, softmin_temp, 
                                        serial, prefix))
    return batch

  def compose(self, cm, f_cv, g_cv, dst_cv, f_z, g_z, dst_z, bbox, 
                    f_mip, g_mip, dst_mip, prefix=''):
    """Compose two vector field CloudVolumes

    For coarse + fine composition:
      f = fine 
      g = coarse 
    
    Args:
       f_cv: MiplessCloudVolume of vector field f
       g_cv: MiplessCloudVolume of vector field g
       dst_cv: MiplessCloudVolume of composed vector field
       f_z: int of section index to process
       g_z: int of section index to process
       dst_z: int of section index to process
       bbox: BoundingBox of region to process
       f_mip: MIP of vector field f
       g_mip: MIP of vector field g
       dst_mip: MIP of composed vector field
       wait: bool indicating whether to wait for all tasks must finish before proceeding
       prefix: str used to write "finished" files for each task 
        (only used for distributed)
    """
    start = time()
    chunks = self.break_into_chunks(bbox, cm.vec_chunk_sizes[dst_mip],
                                    cm.vec_voxel_offsets[dst_mip], 
                                    mip=dst_mip)
    if prefix == '':
      prefix = '{}_{}'.format(dst_mip, dst_z)
    batch = []
    for chunk in chunks:
      batch.append(tasks.ComposeTask(f_cv, g_cv, dst_cv, f_z, g_z, 
                                     dst_z, chunk, f_mip, g_mip, dst_mip,
                                     prefix))
    return batch

  def cpc_chunkwise(self, src_z, tgt_z, src_cv, tgt_cv, dst_cv, bbox, src_mip, dst_mip):
    """Chunkwise CPC 

    Args:
       src_z: int z index of one section to compare
       tgt_z: int z index of other section to compare
       src_cv: MiplessCloudVolume of source image
       tgt_cv: MiplessCloudVolume of target image
       dst_cv: MiplessCloudVolume of destination image
       bbox: BoundingBox of region to process
       src_mip: int MIP level of input src & tgt images
       dst_mip: int MIP level of output image, will dictate the size of the chunks
        used for the pearson r
    """
    
    print('Compute CPC of MIP{0} at MIP{1}, {2}<-({2},{3})'.format(src_mip, dst_mip, 
                                                                   src_z, tgt_z))
    start = time()
    chunks = self.break_into_chunks(bbox, cm.dst_chunk_sizes[dst_mip],
                                    cm.dst_voxel_offsets[dst_mip], mip=dst_mip, 
                                    render=True)
    if self.distributed:
        batch = []
        for i in range(0, len(chunks), self.task_batch_size):
            task_patches = []
            for j in range(i, min(len(chunks), i + self.task_batch_size)):
                task_patches.append(chunks[j].serialize())
            batch.append(tasks.RenderCVTask(src_z, field_cv, field_z, task_patches,
                                                      mip, dst_cv, dst_z))
        self.upload_tasks(batch)
        self.wait_for_queue_empty(dst_cv.path,'render_batch/'+str(mip)+'_'+str(dst_z)+'_'+str(batch)+'/', len(chunks))
    else:
        def chunkwise(patch_bbox):
          r = self.cpc(src_z, tgt_z, src_cv, tgt_cv, patch_bbox, src_mip, dst_mip)
          r = r.cpu().numpy()
          self.save_image(dst_cv, src_z, r, patch_bbox, dst_mip)
        self.pool.map(chunkwise, chunks)
    end = time()
    print (": {} sec".format(end - start))

  def render_batch_chunkwise(self, src_z, field_cv, field_z, dst_cv, dst_z, bbox, mip,
                   batch):
    """Chunkwise render

    Warp the image in BBOX at MIP and SRC_Z in CloudVolume dir at SRC_Z_OFFSET, 
    using the field at FIELD_Z in CloudVolume dir at FIELD_Z_OFFSET, and write 
    the result to DST_Z in CloudVolume dir at DST_Z_OFFSET. Chunk BBOX 
    appropriately.
    """
    
    print('Rendering src_z={0} @ MIP{1} to dst_z={2}'.format(src_z, mip, dst_z), flush=True)
    start = time()
    print("chunk_size: ", cm.dst_chunk_sizes[mip], cm.dst_voxel_offsets[mip])
    chunks = self.break_into_chunks_v2(bbox, cm.dst_chunk_sizes[mip],
                                    cm.dst_voxel_offsets[mip], mip=mip, render=True)
    if self.distributed:
        batch = []
        for i in range(0, len(chunks), self.task_batch_size):
            task_patches = []
            for j in range(i, min(len(chunks), i + self.task_batch_size)):
                task_patches.append(chunks[j].serialize())
            batch.append(tasks.RenderLowMipTask(src_z, field_cv, field_z, 
                                                           task_patches, image_mip, 
                                                           vector_mip, dst_cv, dst_z))
            self.upload_tasks(batch)
        self.wait_for_queue_empty(dst_cv.path, 'render_done/'+str(mip)+'_'+str(dst_z)+'/', len(chunks))
    else:
        def chunkwise(patch_bbox):
          warped_patch = self.cloudsample_image_batch(src_z, field_cv, field_z,
                                                      patch_bbox, mip, batch)
          self.save_image_batch(dst_cv, (dst_z, dst_z + batch), warped_patch, patch_bbox, mip)
        self.pool.map(chunkwise, chunks)
    end = time()
    print (": {} sec".format(end - start))

  def downsample_chunkwise(self, cv, z, bbox, source_mip, target_mip, wait=True):
    """Chunkwise downsample

    For the CloudVolume dirs at Z_OFFSET, warp the SRC_IMG using the FIELD for
    section Z in region BBOX at MIP. Chunk BBOX appropriately and save the result
    to DST_IMG.
    """
    print("Downsampling {} from mip {} to mip {}".format(bbox.__str__(mip=0), source_mip, target_mip))
    for m in range(source_mip+1, target_mip+1):
      chunks = self.break_into_chunks(bbox, cm.dst_chunk_sizes[m],
                                      cm.dst_voxel_offsets[m], mip=m, render=True)
      if self.distributed and len(chunks) > self.task_batch_size * 4:
          batch = []
          print("Distributed downsampling to mip", m, len(chunks)," chunks")
          for i in range(0, len(chunks), self.task_batch_size * 4):
              task_patches = []
              for j in range(i, min(len(chunks), i + self.task_batch_size * 4)):
                  task_patches.append(chunks[j].serialize())
              batch.append(tasks.DownsampleTask(cv, z, task_patches, mip=m))
          self.upload_tasks(batch)
          if wait:
            self.task_queue.block_until_empty()
      else:
          def chunkwise(patch_bbox):
            print ("Local downsampling {} to mip {}".format(patch_bbox.__str__(mip=0), m))
            downsampled_patch = self.downsample_patch(cv, z, patch_bbox, m-1)
            self.save_image_patch(cv, z, downsampled_patch, patch_bbox, m)
          self.pool.map(chunkwise, chunks)

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
      cv_path = self.dst[0].root
      batch = []
      for patch_bbox in chunks:
        batch.append(tasks.ResidualTask(src_z, src_cv, tgt_z, tgt_cv,
                                                  field_cv, patch_bbox, mip,
                                                  cv_path))
      self.upload_tasks(batch)
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
    chunks = self.break_into_chunks(bbox, cm.vec_chunk_sizes[mip],
                                    cm.vec_voxel_offsets[mip], mip=mip)
    print("Vector field inversion for slice {0} @ MIP{1} ({2} chunks)".
           format(z, mip, len(chunks)), flush=True)
    if self.distributed:
        batch = []
        for patch_bbox in chunks:
          batch.append(tasks.InvertFieldTask(z, src_cv, dst_cv, patch_bbox, 
                                                      mip, optimizer))
        self.upload_tasks(batch)
    else: 
    #for patch_bbox in chunks:
        def chunkwise(patch_bbox):
          self.invert_field(z, src_cv, dst_cv, patch_bbox, mip)
        self.pool.map(chunkwise, chunks)
    end = time()
    print (": {} sec".format(end - start))

  def res_and_compose(self, z, forward_match, reverse_match, bbox, mip, write_F_cv):
      tgt_range = []
      T = 2**mip
      if forward_match:
        tgt_range.extend(range(self.tgt_range[-1], 0, -1)) 
      if reverse_match:
        tgt_range.extend(range(self.tgt_range[0], 0, 1)) 
      fields = []
      for z_offset in tgt_range:
          if z_offset != 0:
            src_z = z
            tgt_z = src_z - z_offset
            #print("------------------zoffset is", z_offset)
            f = self.get_residual(src_z, tgt_z, bbox, mip)
            fields.append(f) 
      field = vector_vote(fields, T=T)
      field = field.data.cpu().numpy() 
      self.save_field(write_F_cv, z, field, bbox, mip, relative=False, as_int16=as_int16)

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
    

  def generate_pairwise_and_compose(self, z_range, compose_start, bbox, mip, forward_match,
                                    reverse_match, batch_size=1):
    """Create all pairwise matches for each SRC_Z in Z_RANGE to each TGT_Z in TGT_RADIUS
  
    Args:
        z_range: list of z indices to be matches 
        bbox: BoundingBox object for bounds of 2D region
        forward_match: bool indicating whether to match from z to z-i
          for i in range(tgt_radius)
        reverse_match: bool indicating whether to match from z to z+i
          for i in range(tgt_radius)
        batch_size: (for distributed only) int describing how many sections to issue 
          multi-match tasks for, before waiting for all tasks to complete
    """
    
    m = mip
    batch_count = 0
    start = 0
    chunks = self.break_into_chunks(bbox, cm.vec_chunk_sizes[m],
                                    cm.vec_voxel_offsets[m], mip=m)
    if forward_match:
      cm.add_composed_cv(compose_start, inverse=False,
                                  as_int16=as_int16)
      write_F_k = cm.get_composed_key(compose_start, inverse=False)
      write_F_cv = cm.for_write(write_F_k)
    if reverse_match:
      cm.add_composed_cv(compose_start, inverse=True,
                                  as_int16=as_int16)
      write_invF_k = cm.get_composed_key(compose_start, inverse=True)
      write_F_cv = cm.for_write(write_invF_k)

    for z in z_range:
      start = time()
      batch_count += 1
      i = 0
      if self.distributed:
        print("chunks size is", len(chunks))
        batch = []
        for patch_bbox in chunks:
            batch.append(tasks.ResAndComposeTask(z, forward_match,
                                                        reverse_match,
                                                        patch_bbox, mip,
                                                        write_F_cv))
        self.upload_tasks(batch)
      else:
        def chunkwise(patch_bbox):
            self.res_and_compose(z, forward_match, reverse_match, patch_bbox,
                                mip, write_F_cv)
        self.pool.map(chunkwise, chunks)
      if batch_count == batch_size and self.distributed:
        print('generate_pairwise waiting for {batch} sections'.format(batch=batch_size))
        print('batch_count is {}'.format(batch_count), flush = True)
        self.task_queue.block_until_empty()
        end = time()
        print (": {} sec".format(end - start))
        batch_count = 0
    # report on remaining sections after batch 
    if batch_count > 0 and self.distributed:
      print('generate_pairwise waiting for {batch} sections'.format(batch=batch_size))
      self.task_queue.block_until_empty()
      end = time()
      print (": {} sec".format(end - start))


  def generate_pairwise(self, z_range, bbox, forward_match, reverse_match, 
                              render_match=False, batch_size=1, wait=True):
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
        wait: (for distributed only) bool to wait after batch_size for all tasks
          to finish
    """
    
    mip = self.process_low_mip
    batch_count = 0
    start = 0
    for z in z_range:
      start = time()
      batch_count += 1 
      self.multi_match(z, forward_match=forward_match, reverse_match=reverse_match, 
                       render=render_match)
      if batch_count == batch_size and self.distributed and wait:
        print('generate_pairwise waiting for {batch} section(s)'.format(batch=batch_size))
        self.task_queue.block_until_empty()
        end = time()
        print (": {} sec".format(end - start))
        batch_count = 0
    # report on remaining sections after batch 
    if batch_count > 0 and self.distributed and wait:
      print('generate_pairwise waiting for {batch} section(s)'.format(batch=batch_size))
      self.task_queue.block_until_empty()
    end = time()
    print (": {} sec".format(end - start))
    #if self.p_render:
    #    self.task_queue.block_until_empty()
 
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
    
    T = 2**mip
    print('softmin temp: {0}'.format(T))
    if forward_compose:
      cm.add_composed_cv(compose_start, inverse=False,
                                  as_int16=as_int16)
    if inverse_compose:
      cm.add_composed_cv(compose_start, inverse=True,
                                  as_int16=as_int16)
    write_F_k = cm.get_composed_key(compose_start, inverse=False)
    write_invF_k = cm.get_composed_key(compose_start, inverse=True)
    read_F_k = write_F_k
    read_invF_k = write_invF_k
     
    if forward_compose:
      read_F_cv = cm.for_read(read_F_k)
      write_F_cv = cm.for_write(write_F_k)
      self.vector_vote_chunkwise(z_range, read_F_cv, write_F_cv, bbox, mip, 
                                 inverse=False, T=T, negative_offsets=negative_offsets,
                                 serial_operation=serial_operation)
    if inverse_compose:
      read_F_cv = cm.for_read(read_invF_k)
      write_F_cv = cm.for_write(write_invF_k)
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
      F = self.get_field(F_cv, tgt_z, bbox, mip, relative=True, to_tensor=True,
                        as_int16=as_int16)
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
    next_F = self.get_field(F_cv, next_z, bbox, mip, relative=True,
                            to_tensor=True, as_int16=as_int16)
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
    cm.add_composed_cv(curr_block, inverse=False,
                                as_int16=as_int16)
    cm.add_composed_cv(curr_block, inverse=True,
                                as_int16=as_int16)
    cm.add_composed_cv(next_block, inverse=False,
                                as_int16=as_int16)
    F_cv = cm.get_composed_cv(curr_block, inverse=False, for_read=True)
    invF_cv = cm.get_composed_cv(curr_block, inverse=True, for_read=True)
    next_cv = cm.get_composed_cv(next_block, inverse=False, for_read=False)
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
      F = self.get_field(F_cv, z, bbox, mip, relative=True, to_tensor=True,
                         as_int16=as_int16)
      avg_invF = torch.sum(torch.mul(bump, invFs), dim=0, keepdim=True)
      regF = compose_fields(avg_invF, F)
      regF = regF.data.cpu().numpy() 
      self.save_field(next_cv, z, regF, bbox, mip, relative=True, as_int16=as_int16)
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
    # cm.add_composed_cv(compose_start, inverse=False)
    # cm.add_composed_cv(compose_start, inverse=True)
    chunks = self.break_into_chunks(bbox, cm.vec_chunk_sizes[mip],
                                    cm.vec_voxel_offsets[mip], mip=mip)
    print("Regularizing slice range {0} @ MIP{1} ({2} chunks)".
           format(z_range, mip, len(chunks)), flush=True)
    if self.distributed:
        batch = []
        for patch_bbox in chunks:
            batch.append(tasks.RegularizeTask(z_range[0], z_range[-1],
                                                      dir_z, patch_bbox,
                                                      mip, sigma))
        self.upload_tasks(batch)
        self.task_queue.block_until_empty()
    else:
        #for patch_bbox in chunks:
        def chunkwise(patch_bbox):
          self.regularize_z(z_range, dir_z, patch_bbox, mip, sigma=sigma)
        self.pool.map(chunkwise, chunks)
    end = time()
    print (": {} sec".format(end - start))

  def wait_for_queue_empty(self, path, prefix, chunks_len):
    if self.distributed:
      print("\nWait\n"
            "path {}\n"
            "prefix {}\n"
            "{} chunks\n".format(path, prefix, chunks_len), flush=True)
      empty = False
      n = 0
      while not empty:
        if n > 0:
          sleep(1.75)
        with Storage(path) as stor:
            lst = stor.list_files(prefix=prefix)
        i = sum(1 for _ in lst)
        empty = (i == chunks_len)
        n += 1

  def wait_for_queue_empty_range(self, path, prefix, z_range, chunks_len):
      i = 0
      with Storage(path) as stor:
          for z in z_range:
              lst = stor.list_files(prefix=prefix+str(z))
              i += sum(1 for _ in lst)
      return i == chunks_len
