import torch
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
from skimage import img_as_ubyte
from skimage.filters import gabor
from skimage.morphology import rectangle, dilation, closing, opening
from taskqueue import TaskQueue, LocalTaskQueue
from torch.nn.functional import interpolate, max_pool2d, conv2d
import torch.nn as nn
import torchfields

from normalizer import Normalizer
from scipy.special import binom
from temporal_regularization import create_field_bump
from training.loss import lap
from utilities.helpers import save_chunk, crop, upsample, grid_sample, \
                              np_downsample, invert, compose_fields, upsample_field, downsample_field, \
                              is_identity, cpc, vector_vote, get_affine_field, is_blank, \
                              identity_grid, percentile, coarsen_mask
from boundingbox import BoundingBox, deserialize_bbox

from pathos.multiprocessing import ProcessPool, ThreadPool
from threading import Lock
from pathlib import Path
from utilities.archive import ModelArchive

import torch.nn as nn
#from taskqueue import TaskQueue
import tasks
import tenacity
import boto3
from fcorr import get_fft_power2, get_hp_fcorr

retry = tenacity.retry(
  reraise=True,
  stop=tenacity.stop_after_attempt(7),
  wait=tenacity.wait_full_jitter(0.5, 60.0),
)

class Aligner:
  def __init__(self, threads=1, queue_name=None, task_batch_size=1,
               device='cuda', dry_run=False, completed_queue_name=None, **kwargs):
    print('Creating Aligner object')

    self.distributed = (queue_name != None)
    self.queue_name = queue_name
    self.task_queue = None
    self.sqs = None
    self.queue_url = None
    if queue_name:
      self.task_queue = TaskQueue(queue_name=queue_name, n_threads=0)

    if completed_queue_name is None:
      self.completed_task_queue = None
    else:
      self.completed_task_queue = TaskQueue(queue_name=completed_queue_name, n_threads=0)

    # self.chunk_size = (1024, 1024)
    self.chunk_size = (4096, 4096)
    self.device = torch.device(device)

    self.model_archives = {}

    # self.pool = None #ThreadPool(threads)
    self.threads = threads
    self.task_batch_size = task_batch_size
    self.dry_run = dry_run
    self.eps = 1e-6

    self.gpu_lock = kwargs.get('gpu_lock', None)  # multiprocessing.Semaphore

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
      new_bbox = BoundingBox(x_range[0] + dis[0], x_range[1] + dis[0],
                             y_range[0] + dis[1], y_range[1] + dis[1],
                             mip=0)
      new_bbox.max_mip = bbox.max_mip
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
      print('Loading model {0} from cache'.format(model_path), flush=True)
      return self.model_archives[model_path]
    else:
      print('Adding model {0} to the cache'.format(model_path), flush=True)
      path = Path(model_path)
      model_name = path.stem
      archive = ModelArchive(model_name)
      self.model_archives[model_path] = archive
      return archive

  #######################
  # Image IO + handlers #
  #######################
  def get_masks(self, masks, z, bbox, dst_mip, to_tensor=True,
               mask_op='none'):
        start = time()
        result = None
        for mask in masks:
            mask_data = self.get_mask(mask.cv, z, bbox, mask.mip, dst_mip, mask.val,
                                to_tensor=to_tensor, mask_op=mask_op,
                                coarsen_count=mask.coarsen_count).long()
            if result is None:
                result = mask_data
            else:
                result[mask_data > 0] = mask_data[mask_data > 0]

        end = time()
        diff = end - start
        print('get_masks: {:.3f}'.format(diff), flush=True)
        return result


  def get_mask(self, cv, z, bbox, src_mip, dst_mip, valid_val, to_tensor=True,
               mask_op='none', coarsen_count=0):
    start = time()
    data = self.get_data(cv, z, bbox, src_mip=src_mip, dst_mip=dst_mip,
                             to_float=False, to_tensor=to_tensor, normalizer=None)
    if mask_op == 'eq':
        mask = data == valid_val
    elif mask_op == 'lt':
        mask = data < valid_val
    elif mask_op == 'gt':
        mask = data > valid_val
    elif mask_op == 'lte':
        mask = data <= valid_val
    elif mask_op == 'gte':
        mask = data >= valid_val
    elif mask_op == 'ne':
        mask = data != valid_val
    elif mask_op == 'none':
        mask = data != data
    elif mask_op == 'data':
        mask = data
    else:
        raise Exception("Mask op {} unsupported".format(mask_op))
    if coarsen_count > 0:
        mask = coarsen_mask(mask, count=coarsen_count)
    end = time()
    diff = end - start
    print('get_mask: {:.3f}'.format(diff), flush=True)
    return mask

  def get_image(self, cv, z, bbox, mip, to_tensor=True, normalizer=None,
                dst_mip=None):
    print('get_image for {0}'.format(bbox.stringify(z)), flush=True)
    start = time()
    if dst_mip == None:
        d_mip = mip
    else:
        d_mip = dst_mip
    image = self.get_data(cv, z, bbox, src_mip=mip, dst_mip=d_mip, to_float=True,
                             to_tensor=to_tensor, normalizer=normalizer)
    end = time()
    diff = end - start
    print('get_image: {:.3f}'.format(diff), flush=True)
    return image

  def get_masked_image(self, image_cv, z, bbox, image_mip, masks,
                             to_tensor=True, normalizer=None, mask_op='none',
                             return_mask=False, blackout=True):
    """Get image with mask applied
    """
    start = time()
    image = self.get_image(image_cv, z, bbox, image_mip,
                           to_tensor=True, normalizer=normalizer)
    if len(masks) > 0:
      mask = self.get_masks(masks, z, bbox,
                           dst_mip=image_mip, mask_op=mask_op
                           )
      if blackout:
          image = image.masked_fill_(mask > 0, 0)

    if not to_tensor:
      image = image.cpu().numpy()
      mask  = mask.cpu().numpy()

    end = time()
    diff = end - start
    print('get_masked_image: {:.3f}'.format(diff), flush=True)
    if return_mask:
        return image, mask
    else:
        return image

  def get_composite_image(self, image_cv, z_list, bbox, image_mip,
                                masks=[],
                                to_tensor=True, normalizer=None,
                                mask_op='none'):
    """Collapse a stack of 2D image into a single 2D image, by consecutively
        replacing black pixels (0) in the image of the first z_list entry with
        non-black pixels from of the consecutive z_list entries images.

    Args:
       image_cv: MiplessCloudVolume where images are stored
       z_list: list of image indices processed in the given order
       bbox: BoundingBox defining data range
       image_mip: int MIP level of the image data to process
       mask_cv: MiplessCloudVolume where masks are stored, or None if no mask
        should be used
       mask_mip: int MIP level of the mask, ignored if ``mask_cv`` is None
       mask_val: The mask value that specifies regions to be blackened, ignored
        if ``mask_cv`` is None.
       to_tensor: output will be torch.tensor
       #TODO normalizer: callable function to adjust the contrast of each image
    """

    # Retrieve image stack
    assert len(z_list) > 0

    combined = self.get_masked_image(image_cv, z_list[0], bbox, image_mip,
                                     masks,
                                     to_tensor=to_tensor, normalizer=normalizer,
                                     mask_op=mask_op)
    for z in z_list[1:]:
      tmp = self.get_masked_image(image_cv, z, bbox, image_mip,
                                  masks,
                                  to_tensor=to_tensor, normalizer=normalizer,
                                   mask_op=mask_op)
      black_mask = combined == 0
      combined[black_mask] = tmp[black_mask]

    return combined

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
      if data.dtype == np.uint8:
        data = np.divide(data, float(255.0), dtype=np.float32)
    if (normalizer is not None) and (not is_blank(data)):
      print('Normalizing image')
      start = time()
      data = torch.from_numpy(data)
      data = data.to(device=self.device)
      data = normalizer(data).reshape(data.shape)
      end = time()
      diff = end - start
      print('normalizer: {:.3f}'.format(diff), flush=True)
    # convert to tensor if requested, or if up/downsampling required
    if to_tensor | (src_mip != dst_mip):
      if isinstance(data, np.ndarray):
        if (data.dtype == np.uint32):
          data = data.astype(np.int64)
        data = torch.from_numpy(data)
      if self.device.type == 'cuda':
        data = data.to(device=self.device)
        if src_mip != dst_mip:
          # k = 2**(src_mip - dst_mip)
          size = (bbox.y_size(dst_mip), bbox.x_size(dst_mip))
          if not isinstance(data, torch.cuda.ByteTensor) and not isinstance(data, torch.cuda.LongTensor):
            data = interpolate(data, size=size, mode='bilinear')
          else:
            data = data.type('torch.cuda.DoubleTensor')
            data = interpolate(data, size=size, mode='nearest')
            data = data.type('torch.cuda.ByteTensor')
      else:
        data = data.float()
        if src_mip > dst_mip:
          size = (bbox.y_size(dst_mip), bbox.x_size(dst_mip))
          data = interpolate(data, size=size, mode='nearest')
          data = data.type(torch.ByteTensor)
        elif src_mip < dst_mip:
          ratio = 2**(dst_mip-src_mip)
          data = max_pool2d(data, kernel_size=ratio)
          data = data.type(torch.ByteTensor)
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
    if to_uint8 and cv[mip].dtype != np.float32:
      patch = (np.multiply(patch, 255)).astype(np.uint8)
    # try:
    cv[mip][x_range[0]:x_range[1], y_range[0]:y_range[1], z] = patch
    # except:
      # import ipdb
      # ipdb.set_trace()

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

  def append_image(self, float_patch, cv, z, bbox, mip, to_uint8=True):
    x_range = bbox.x_range(mip=mip)
    y_range = bbox.y_range(mip=mip)
    patch = np.transpose(float_patch, (2,3,0,1))
    #print("----------------z is", z, "save image patch at mip", mip, "range", x_range, y_range, "range at mip0", bbox.x_range(mip=0), bbox.y_range(mip=0))
    if to_uint8:
      patch = (np.multiply(patch, 255)).astype(np.uint8)
    cv[mip][x_range[0]:x_range[1], y_range[0]:y_range[1], z] = cv[mip][x_range[0]:x_range[1], y_range[0]:y_range[1], z] + patch

  def append_image_batch(self, cv, z_range, float_patch, bbox, mip, to_uint8=True):
    x_range = bbox.x_range(mip=mip)
    y_range = bbox.y_range(mip=mip)
    print("type of float_patch", type(float_patch), "shape", float_patch.shape)
    patch = np.transpose(float_patch, (2,3,0,1))
    # patch = np.transpose(float_patch, (2,1,0))[..., np.newaxis]
    if to_uint8:
        patch = (np.multiply(patch, 255)).astype(np.uint8)
    print("patch shape", patch.shape)
    cv[mip][x_range[0]:x_range[1], y_range[0]:y_range[1], z_range[0]:z_range[1]] = cv[mip][x_range[0]:x_range[1], y_range[0]:y_range[1], z_range[0]:z_range[1]] + patch
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
    # if as_int16:
    if cv.dtype == 'int16':
      field = np.float32(field) / 4
    if relative:
      field = self.abs_to_rel_residual(field, bbox, mip)
    if to_tensor:
      field = torch.from_numpy(field)
      return field.to(device=self.device)
    else:
      return field

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
    # if as_int16:
    if cv.dtype == 'int16':
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
    favg = field.sum() / (torch.nonzero(field).size(0) + self.eps)
    return favg

  def profile_field(self, field):
    nonzero = field[(field[...,0] != 0) & (field[...,1] != 0)]
    if len(nonzero) == 0:
      return torch.Tensor([0, 0])

    low_l = percentile(nonzero, 25)
    high_l = percentile(nonzero, 75)
    mid = 0.5*(low_l + high_l)

    print("MID:", mid[0].item(), mid[1].item())
    return mid.cpu()

  #############################
  # CloudVolume chunk methods #
  #############################

  def compute_field_chunk_stitch(self, model_path, src_cv, tgt_cv, src_z, tgt_z, bbox, mip, pad,
                          src_masks=[], tgt_masks=[],
                          tgt_alt_z=None, prev_field_cv=None, prev_field_z=None,
                          prev_field_inverse=False):
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
      prev_field_cv: if specified, a MiplessCloudVolume containing the
                     previously predicted field to be profile and displace
                     the src chunk
    Returns:
      field with MIP0 residuals with the shape of bbox at MIP mip (np.ndarray)
    """
    archive = self.get_model_archive(model_path)
    model = archive.model
    normalizer = archive.preprocessor
    print('compute_field for {0} to {1}'.format(bbox.stringify(src_z),
                                                bbox.stringify(tgt_z)))
    print('pad: {}'.format(pad))
    padded_bbox = deepcopy(bbox)
    padded_bbox.max_mip = mip
    padded_bbox.uncrop(pad, mip=mip)

    if prev_field_cv is not None:
        field = self.get_field(prev_field_cv, prev_field_z, padded_bbox, mip,
                           relative=False, to_tensor=True)
        if prev_field_inverse:
          field = -field
        distance = self.profile_field(field)
        print('Displacement adjustment: {} px'.format(distance))
        distance = (distance // (2 ** mip)) * 2 ** mip
        new_bbox = self.adjust_bbox(padded_bbox, distance.flip(0))
    else:
        distance = torch.Tensor([0, 0])
        new_bbox = padded_bbox

    tgt_z = [tgt_z]
    if tgt_alt_z is not None:
      try:
        tgt_z.extend(tgt_alt_z)
      except TypeError:
        tgt_z.append(tgt_alt_z)
      print('alternative target slices:', tgt_alt_z)

    src_patch = self.get_masked_image(src_cv, src_z, new_bbox, mip,
                                masks=src_masks,
                                to_tensor=True, normalizer=normalizer)
    tgt_patch = self.get_composite_image(tgt_cv, tgt_z, padded_bbox, mip,
                                masks=[],
                                to_tensor=True, normalizer=normalizer)
    print('src_patch.shape {}'.format(src_patch.shape))
    print('tgt_patch.shape {}'.format(tgt_patch.shape))

    # Running the model is the only part that will increase memory consumption
    # significantly - only incrementing the GPU lock here should be sufficient.
    if self.gpu_lock is not None:
      self.gpu_lock.acquire()
      print("Process {} acquired GPU lock".format(os.getpid()))

    try:
      print("GPU memory allocated: {}, cached: {}".format(torch.cuda.memory_allocated(), torch.cuda.memory_cached()))
      zero_fieldC = torch.zeros([1, src_patch.size()[2], src_patch.size()[3], 2], dtype=torch.float32, device=self.device)
      # import ipdb
      # ipdb.set_trace()

      # zero_fieldC = torch.Field(torch.zeros(torch.Size([1,2,2048,2048])))
      # zero_fieldC = zero_fieldC.permute(0,2,3,1).to(device=self.device)

      # model produces field in relative coordinates
      field = model(
        src_patch,
        tgt_patch,
        tgt_field=zero_fieldC,
        src_field=zero_fieldC,
      )

      if not isinstance(field, torch.Tensor):
          field = field[0]

      print("GPU memory allocated: {}, cached: {}".format(torch.cuda.memory_allocated(), torch.cuda.memory_cached()))
      field = self.rel_to_abs_residual(field, mip)
      field = field[:,pad:-pad,pad:-pad,:]
      field += distance.to(device=self.device)
      field = field.data.cpu().numpy()
      # clear unused, cached memory so that other processes can allocate it
      torch.cuda.empty_cache()

      print("GPU memory allocated: {}, cached: {}".format(torch.cuda.memory_allocated(), torch.cuda.memory_cached()))
    finally:
      if self.gpu_lock is not None:
        print("Process {} releasing GPU lock".format(os.getpid()))
        self.gpu_lock.release()

    return field

  def compute_field_chunk(
    self,
    model_path,
    *,
    bbox,
    pad,
    src_cv,
    src_z,
    tgt_cv,
    tgt_z,
    mip,
    src_masks=[],
    tgt_masks=[],
    tgt_alt_z=None,
    prev_field_cv=None,
    prev_field_z=None,
    coarse_field_cv=None,
    coarse_field_mip=None,
    tgt_field_cv=None
  ):
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
      prev_field_cv: if specified, a MiplessCloudVolume containing the
                     previously predicted field to be profile and displace
                     the src chunk

    Returns:
      field with MIP0 residuals with the shape of bbox at MIP mip (np.ndarray)
    """
    archive = self.get_model_archive(model_path)
    model = archive.model
    normalizer = archive.preprocessor
    print(
      "compute_field for {0} to {1}".format(
        bbox.stringify(src_z), bbox.stringify(tgt_z)
      )
    )
    print("pad: {}".format(pad))

    if coarse_field_mip is None:
      coarse_field_mip = bbox.max_mip

    # Find the target patch (Coarse+Fine vector field)
    coarse_field = None
    coarse_distance_fine_snap = torch.Tensor([0, 0])
    drift_distance_fine_snap = torch.Tensor([0, 0])
    drift_distance_coarse_snap = torch.Tensor([0, 0])

    tgt_field = None
    padded_tgt_bbox_fine = deepcopy(bbox)
    padded_tgt_bbox_fine.uncrop(pad, mip)
    tgt_field = None
    if tgt_field_cv is not None:
      # Fetch vector field of target section
      tgt_field = self.get_field(
        tgt_field_cv,
        tgt_z,
        padded_tgt_bbox_fine,
        mip,
        relative=True,
        to_tensor=True,
      ).to(device=self.device)
      #HACKS
      tgt_field = torch.zeros_like(tgt_field)

      if coarse_field_cv is not None and not is_identity(tgt_field):
        # Alignment with coarse field: Need to subtract the coarse field out of
        # the tgt_field to get the current alignment drift
        tgt_coarse_field = self.get_field(
          coarse_field_cv,
          tgt_z,
          padded_tgt_bbox_fine,
          coarse_field_mip,
          relative=True,
          to_tensor=True,
        ).to(device=self.device)

        #HACKS
        tgt_coarse_field = torch.zeros_like(tgt_coarse_field)

        tgt_coarse_field = tgt_coarse_field.permute(0, 3, 1, 2).field_()
        tgt_coarse_field_inv = tgt_coarse_field.inverse().up(coarse_field_mip - mip)

        tgt_drift_field = tgt_coarse_field_inv.compose_with(tgt_field.permute(0, 3, 1, 2).field_())
        tgt_drift_field = tgt_drift_field.permute(0, 2, 3, 1)
      else:
        # Alignment without coarse field: tgt_field contains only the drift
        # or prev_field is identity
        tgt_drift_field = tgt_field


      tgt_drift_field = self.rel_to_abs_residual(tgt_drift_field, mip)
      drift_distance = self.profile_field(tgt_drift_field)
      drift_distance_fine_snap = (drift_distance // (2 ** mip)) * 2 ** mip
      drift_distance_coarse_snap = (
        drift_distance // (2 ** coarse_field_mip)
      ) * 2 ** coarse_field_mip

      tgt_field = self.rel_to_abs_residual(tgt_field, mip)
      tgt_distance = self.profile_field(tgt_field)
      tgt_distance_fine_snap = (tgt_distance // (2 ** mip)) * 2 ** mip
      tgt_field -= tgt_distance_fine_snap.to(device=self.device)
      tgt_field = self.abs_to_rel_residual(tgt_field, padded_tgt_bbox_fine, mip)
      tgt_field = torch.zeros_like(tgt_field)

    print(
      "Displacement adjustment TGT: {} px".format(
        drift_distance_fine_snap,
      )
    )

    padded_tgt_bbox_fine = self.adjust_bbox(padded_tgt_bbox_fine, tgt_distance_fine_snap.flip(0))
    # padded_tgt_bbox_fine.uncrop(pad, mip)

    if coarse_field_cv is not None:
      # Fetch coarse alignment
      padded_src_bbox_coarse = self.adjust_bbox(bbox, drift_distance_coarse_snap.flip(0))
      padded_src_bbox_coarse.max_mip = coarse_field_mip
      padded_src_bbox_coarse.uncrop(pad, mip)

      padded_src_bbox_coarse_field = deepcopy(padded_src_bbox_coarse)
      padded_src_bbox_coarse_field.uncrop(2**coarse_field_mip, 0)

      coarse_field = self.get_field(
        coarse_field_cv,
        src_z,
        padded_src_bbox_coarse_field,
        coarse_field_mip,
        relative=False,
        to_tensor=True,
      ).to(device=self.device)
      coarse_field = coarse_field.permute(0, 3, 1, 2)
      # coarse_field = nn.Upsample(scale_factor=2**(coarse_field_mip - mip), mode='bilinear')(coarse_field)
      coarse_field = nn.Upsample(scale_factor=2**(coarse_field_mip - mip), mode='bicubic')(coarse_field)
      coarse_field = coarse_field.permute(0, 2, 3, 1)

      crop = 2 ** (coarse_field_mip - mip)
      offset = np.array((drift_distance_fine_snap - drift_distance_coarse_snap) // 2 ** mip, dtype=np.int)
      coarse_field = coarse_field[:, crop+offset[1]:-crop+offset[1], crop+offset[0]:-crop+offset[0], :]

      coarse_distance = self.profile_field(coarse_field)
      coarse_distance_fine_snap = (coarse_distance // (2 ** mip)) * 2 ** mip
      coarse_field -= coarse_distance_fine_snap.to(device=self.device)
      coarse_field = self.abs_to_rel_residual(coarse_field, padded_src_bbox_coarse, mip)

    combined_distance_fine_snap = (
      coarse_distance_fine_snap + drift_distance_fine_snap
    )

    print(
      "Displacement adjustment SRC: {} + {} --> {} px".format(
        coarse_distance_fine_snap,
        drift_distance_fine_snap,
        combined_distance_fine_snap,
      )
    )

    padded_src_bbox_fine = self.adjust_bbox(bbox, combined_distance_fine_snap.flip(0))
    padded_src_bbox_fine.uncrop(pad, mip)

    tgt_z = [tgt_z]
    if tgt_alt_z is not None:
      try:
        tgt_z.extend(tgt_alt_z)
      except TypeError:
        tgt_z.append(tgt_alt_z)
      print("alternative target slices:", tgt_alt_z)

    src_patch, src_mask = self.get_masked_image(
      src_cv,
      src_z,
      padded_src_bbox_fine,
      mip,
      masks=src_masks,
      to_tensor=True,
      normalizer=normalizer,
      mask_op='data',
      return_mask=True,
      blackout=False
    )


    padded_tgt_bbox_fine = deepcopy(bbox)
    padded_tgt_bbox_fine.uncrop(pad, mip)
    tgt_patch = self.get_composite_image(
      tgt_cv,
      tgt_z,
      padded_tgt_bbox_fine,
      mip,
      masks=[],
      to_tensor=True,
      normalizer=normalizer,
    )
    print("src_patch.shape {}".format(src_patch.shape))
    print("tgt_patch.shape {}".format(tgt_patch.shape))

    # Running the model is the only part that will increase memory consumption
    # significantly - only incrementing the GPU lock here should be sufficient.
    if self.gpu_lock is not None:
      start = time.time()
      self.gpu_lock.acquire()
      end = time.time()
      print("Process {} acquired GPU lock. Locked time: {0:.2f} sec".format(os.getpid(), end - start))

    try:
      print(
        "GPU memory allocated: {}, cached: {}".format(
          torch.cuda.memory_allocated(), torch.cuda.memory_cached()
        )
      )

      # model produces field in relative coordinates
      accum_field = model(
        src_patch,
        tgt_patch,
        tgt_field=tgt_field,
        src_field=coarse_field,
        src_mask=src_mask
      )

      if not isinstance(accum_field, torch.Tensor):
          accum_field = accum_field[0]

      print(
        "GPU memory allocated: {}, cached: {}".format(
          torch.cuda.memory_allocated(), torch.cuda.memory_cached()
        )
      )

      accum_field = self.rel_to_abs_residual(accum_field, mip)
      accum_field = accum_field[:, pad:-pad, pad:-pad, :]
      accum_field += combined_distance_fine_snap.to(device=self.device)
      accum_field = accum_field.data.cpu().numpy()

      # clear unused, cached memory so that other processes can allocate it
      torch.cuda.empty_cache()

      print(
        "GPU memory allocated: {}, cached: {}".format(
          torch.cuda.memory_allocated(), torch.cuda.memory_cached()
        )
      )
    finally:
      if self.gpu_lock is not None:
        print("Process {} releasing GPU lock".format(os.getpid()))
        self.gpu_lock.release()
    return accum_field

  def predict_image(self, cm, model_path, src_cv, dst_cv, z, mip, bbox,
                    chunk_size):
    start = time()
    chunks = self.break_into_chunks(bbox, chunk_size,
                                    cm.dst_voxel_offsets[mip], mip=mip,
                                    max_mip=cm.num_scales)
    print("\nfold detect\n"
          "model {}\n"
          "src {}\n"
          "dst {}\n"
          "z={} \n"
          "MIP{}\n"
          "{} chunks\n".format(model_path, src_cv, dst_cv, z,
                               mip, len(chunks)), flush=True)
    batch = []
    for patch_bbox in chunks:
      batch.append(tasks.PredictImgTask(model_path, src_cv, dst_cv, z, mip,
                                        patch_bbox))
    return batch

  def predict_image_chunk(self, model_path, src_cv, z, mip, bbox):
    archive = self.get_model_archive(model_path, readonly=2)
    model = archive.model
    image = self.get_image(src_cv, z, bbox, mip, to_tensor=True)
    new_image = model(image)
    return new_image


  def vector_vote_chunk(self, pairwise_cvs, vvote_cv, z, bbox, mip,
                        inverse=False, serial=True, softmin_temp=None,
                        blur_sigma=None):
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
       softmin_temp: temperature to use for the softmin in vector voting; default None
        will use formula based on MIP level
       blur_sigma: std dev of Gaussian kernel by which to blur the vector vote inputs;
        default None means no blurring

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
    if len(fields) == 1:
      return fields[0]
    # assign weight w if the difference between majority vector similarities are d
    if not softmin_temp:
      w = 0.99
      d = 2**mip
      n = len(fields)
      m = int(binom(n, (n+1)//2)) - 1
      softmin_temp = 2**mip
    return vector_vote(fields, softmin_temp=softmin_temp, blur_sigma=blur_sigma)

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
                        masks=[], affine=None,
                        use_cpu=False, pad=256, return_mask=False,
                        blackout_mask_op='eq', return_mask_op='eq', as_int16=True):
      """Wrapper for torch.nn.functional.gridsample for CloudVolume image objects

      Args:
        z: int for section index to warp
        image_cv: MiplessCloudVolume storing the image
        field_cv: MiplessCloudVolume storing the vector field
        bbox: BoundingBox for output region to be warped
        image_mip: int for MIP of the image
        field_mip: int for MIP of the vector field; must be >= image_mip.
         If field_mip > image_mip, the field will be upsampled.
        aff: 2x3 ndarray defining affine transform at MIP0 with which to precondition
         the field. If None, then will be ignored (treated as the identity).

      Returns:
        warped image with shape of bbox at MIP image_mip
      """
      if use_cpu:
          self.device = 'cpu'
      # pad = 256
      # pad = 2048
      padded_bbox = deepcopy(bbox)
      print('Padding by {} at MIP{}'.format(pad, image_mip))
      padded_bbox.uncrop(pad, mip=image_mip)

      # Load initial vector field

      if field_cv is not None:
        # assert(field_mip >= image_mip)
        field = self.get_field(field_cv, field_z, padded_bbox, field_mip,
                                 relative=False, to_tensor=True)
        if field_mip > image_mip:
          field = upsample_field(field, field_mip, image_mip)
        elif field_mip < image_mip:
          field = downsample_field(field, field_mip, image_mip)
      else:
        #this is a bit slow
        image = self.get_image(image_cv, image_z, padded_bbox, image_mip,
                               to_tensor=True, normalizer=None)
        field = torch.zeros((image.shape[1], image.shape[2], image.shape[3], 2),
                device=image.device)

      if affine is not None:
        # PyTorch conventions are column, row order (y, then x) so flip
        # the affine matrix and offset
        affine = torch.Tensor(affine).to(field.device)
        affine = affine.flip(0)[:, [1, 0, 2]]  # flip x and y
        offset_y, offset_x = padded_bbox.get_offset(mip=0)

        ident = self.rel_to_abs_residual(
            identity_grid(field.shape, device=field.device), image_mip)

        field += ident
        field[..., 0] += offset_x
        field[..., 1] += offset_y
        field = torch.tensordot(
            affine[:, 0:2], field, dims=([1], [3])).permute(1, 2, 3, 0)
        field[..., :] += affine[:, 2]
        field[..., 0] -= offset_x
        field[..., 1] -= offset_y
        field -= ident

      if is_identity(field):
        image = self.get_image(image_cv, image_z, bbox, image_mip,
                               to_tensor=True, normalizer=None)
        mask = self.get_masks(masks, image_z, bbox,
                               dst_mip=image_mip)
        # image = image.masked_fill_(mask, 0)
        new_bbox = padded_bbox
      else:
        distance = self.profile_field(field)
        distance = (distance // (2 ** image_mip)) * 2 ** image_mip
        new_bbox = self.adjust_bbox(padded_bbox, distance.flip(0))

        field -= distance.to(device = self.device)
        field = self.abs_to_rel_residual(field, padded_bbox, image_mip)
        field = field.to(device = self.device)

        image = self.get_masked_image(image_cv, image_z, new_bbox, image_mip,
                                      masks=masks,
                                      to_tensor=True, normalizer=None,
                                      mask_op=blackout_mask_op)
        image = grid_sample(image, field, padding_mode='zeros')
        image = image[:,:,pad:-pad,pad:-pad]

      if return_mask:
          mask = self.get_masks(masks, image_z, new_bbox,
                                dst_mip=image_mip, mask_op=return_mask_op)
          if mask is not None:
              warped_mask = grid_sample(mask.float(), field, padding_mode='zeros')
              cropped_warped_mask = warped_mask[:,:,pad:-pad,pad:-pad]
          else:
              cropped_warped_mask = None
          return image, cropped_warped_mask
      else:
          return image


  def cloudsample_compose(self, f_cv, g_cv, f_z, g_z, bbox, f_mip, g_mip,
                          dst_mip, factor=1., affine=None, pad=256):
      """Wrapper for torch.nn.functional.gridsample for CloudVolume field objects.

      Gridsampling a field is a composition, such that f(g(x)).

      Args:
         f_cv: MiplessCloudVolume storing the vector field to do the warping
         g_cv: MiplessCloudVolume storing the vector field to be warped
         bbox: BoundingBox for output region to be warped
         f_z, g_z: int for section index from which to read fields
         f_mip, g_mip: int for MIPs of the input fields
         dst_mip: int for MIP of the desired output field
         factor: float to multiply the f vector field by
         affine: an additional affine matrix to be composed before the fields
           If a is the affine matrix, then rendering the resulting field would
           be equivalent to
             f(g(a(x)))
         pad: number of pixels to pad at dst_mip

      Returns:
         composed field
      """
      assert(f_mip >= dst_mip)
      assert(g_mip >= dst_mip)
      padded_bbox = deepcopy(bbox)
      padded_bbox.max_mip = max(dst_mip, f_mip, g_mip)
      print('Padding by {} at MIP{}'.format(pad, dst_mip))
      padded_bbox.uncrop(pad, mip=dst_mip)
      # Load warper vector field

      f = self.get_field(f_cv, f_z, padded_bbox, f_mip,
                             relative=False, to_tensor=True)
      f = f * factor
      if f_mip > dst_mip:
        f = upsample_field(f, f_mip, dst_mip)

      if is_identity(f):
        g = self.get_field(g_cv, g_z, padded_bbox, g_mip,
                           relative=False, to_tensor=True)
        if g_mip > dst_mip:
            g = upsample_field(g, g_mip, dst_mip)
        return g[:,pad:-pad,pad:-pad,:]

      distance = self.profile_field(f)
      distance = (distance // (2 ** g_mip)) * 2 ** g_mip
      new_bbox = self.adjust_bbox(padded_bbox, distance.flip(0))

      f -= distance.to(device = self.device)
      f = self.abs_to_rel_residual(f, padded_bbox, dst_mip)
      f = f.to(device = self.device)

      g = self.get_field(g_cv, g_z, new_bbox, g_mip,
                         relative=False, to_tensor=True)
      if g_mip > dst_mip:
        g = upsample_field(g, g_mip, dst_mip)
      g = self.abs_to_rel_residual(g, padded_bbox, dst_mip)
      h = compose_fields(f, g)
      h = self.rel_to_abs_residual(h, dst_mip)
      h += distance.to(device=self.device)
      h = h[:,pad:-pad,pad:-pad,:]

      if affine is not None:
        # PyTorch conventions are column, row order (y, then x) so flip
        # the affine matrix and offset
        affine = torch.Tensor(affine).to(f.device)
        affine = affine.flip(0)[:, [1, 0, 2]]  # flip x and y
        offset_y, offset_x = padded_bbox.get_offset(mip=0)

        ident = self.rel_to_abs_residual(
            identity_grid(f.shape, device=f.device), dst_mip)

        h += ident
        h[..., 0] += offset_x
        h[..., 1] += offset_y
        h = torch.tensordot(
            affine[:, 0:2], h, dims=([1], [3])).permute(1, 2, 3, 0)
        h[..., :] += affine[:, 2]
        h[..., 0] -= offset_x
        h[..., 1] -= offset_y
        h -= ident

      return h

  def cloudsample_multi_compose(self, field_list, z_list, bbox, mip_list,
                                dst_mip, factors=None, pad=256):
    """Compose a list of field CloudVolumes

    This takes a list of fields
    field_list = [f_0, f_1, ..., f_n]
    and composes them to get
    f_0 ⚬ f_1 ⚬ ... ⚬ f_n ~= f_0(f_1(...(f_n)))

    Args:
       field_list: list of MiplessCloudVolume storing the vector fields
       z_list: int or list of ints for section indices to read fields
       bbox: BoundingBox for output region to be warped
       mip_list: int or list of ints for MIPs of the input fields
       dst_mip: int for MIP of the desired output field
       pad: number of pixels to pad at dst_mip
       factors: floats to multiply/reweight the fields by before composing

    Returns:
       composed field
    """
    if isinstance(z_list, int):
        z_list = [z_list] * len(field_list)
    else:
        assert(len(z_list) == len(field_list))
    if isinstance(mip_list, int):
        mip_list = [mip_list] * len(field_list)
    else:
        assert(len(mip_list) == len(field_list))
    assert(min(mip_list) >= dst_mip)
    if factors is None:
        factors = [1.0] * len(field_list)
    else:
        assert(len(factors) == len(field_list))
    padded_bbox = deepcopy(bbox)
    padded_bbox.max_mip = dst_mip
    print('Padding by {} at MIP{}'.format(pad, dst_mip))
    padded_bbox.uncrop(pad, mip=dst_mip)

    # load the first vector field
    f_cv, *field_list = field_list
    f_z, *z_list = z_list
    f_mip, *mip_list = mip_list
    f_factor, *factors = factors
    f = self.get_field(f_cv, f_z, padded_bbox, f_mip,
                       relative=False, to_tensor=True)
    f = f * f_factor
    if len(field_list) == 0:
        return f[:, pad:-pad, pad:-pad, :]

    # skip any empty / identity fields
    while is_identity(f):
        f_cv, *field_list = field_list
        f_z, *z_list = z_list
        f_mip, *mip_list = mip_list
        f_factor, *factors = factors
        f = self.get_field(f_cv, f_z, padded_bbox, f_mip,
                           relative=False, to_tensor=True)
        f = f * f_factor
        if len(field_list) == 0:
            return f[:, pad:-pad, pad:-pad, :]

    if f_mip > dst_mip:
        f = upsample_field(f, f_mip, dst_mip)

    # compose with the remaining fields
    while len(field_list) > 0:
        g_cv, *field_list = field_list
        g_z, *z_list = z_list
        g_mip, *mip_list = mip_list
        g_factor, *factors = factors

        distance = self.profile_field(f)
        distance = (distance // (2 ** g_mip)) * 2 ** g_mip
        new_bbox = self.adjust_bbox(padded_bbox, distance.flip(0))

        f -= distance.to(device=self.device)
        f = self.abs_to_rel_residual(f, padded_bbox, dst_mip)
        f = f.to(device=self.device)

        g = self.get_field(g_cv, g_z, new_bbox, g_mip,
                           relative=False, to_tensor=True)
        g = g * g_factor
        if g_mip > dst_mip:
            g = upsample_field(g, g_mip, dst_mip)
        g = self.abs_to_rel_residual(g, padded_bbox, dst_mip)
        h = compose_fields(f, g)
        h = self.rel_to_abs_residual(h, dst_mip)
        h += distance.to(device=self.device)
        f = h
    return f[:, pad:-pad, pad:-pad, :]

  def cloudsample_image_batch(self, z_range, image_cv, field_cv,
                              bbox, image_mip, field_mip,
                              masks=[],
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
                                  masks=masks,
                                  as_int16=as_int16)
      batch.append(image)
    return torch.cat(batch, axis=0)

  def downsample(self, cv, z, bbox, mip):
    data = self.get_image(cv, z, bbox, mip, adjust_contrast=False, to_tensor=True)
    data = interpolate(data, scale_factor=0.5, mode='bilinear')
    return data.cpu().numpy()

  def cpc_chunk(self, src_cv, tgt_cv, src_z, tgt_z, bbox, src_mip, dst_mip, norm=True):
    """Calculate the chunked pearson r between two chunks

    Args:
       src_cv: MiplessCloudVolume of source image
       tgt_cv: MiplessCloudVolume of target image
       src_z: int z index of one section to compare
       tgt_z: int z index of other section to compare
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
    src = self.get_image(src_cv, src_z, bbox, src_mip, normalizer=None,
                         to_tensor=True)
    tgt = self.get_image(tgt_cv, tgt_z, bbox, src_mip, normalizer=None,
                         to_tensor=True)
    print('src.shape {}'.format(src.shape))
    print('tgt.shape {}'.format(tgt.shape))
    return cpc(src, tgt, scale_factor, norm=norm, device=self.device)

  ######################
  # Dataset operations #
  ######################
  def copy(self, cm, src_cv, dst_cv, src_z, dst_z, bbox, mip, is_field=False,
           to_uint8=False, masks=[],
           return_iterator=False):
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
       is_field: bool indicating whether this is a field CloudVolume to copy
       to_uint8: bool indicating whether this image should be saved as float
       mask_cv: MiplessCloudVolume where source mask is stored
       mask_mip: int for MIP level at which source mask is stored
       mask_val: int for pixel value in the mask that should be zero-filled

    Returns:
       a list of CopyTasks
    """
    class CopyTaskIterator():
        def __init__(self, cl, start, stop):
          self.chunklist = cl
          self.start = start
          self.stop = stop
        def __len__(self):
          return self.stop - self.start
        def __getitem__(self, slc):
          itr = deepcopy(self)
          itr.start = slc.start
          itr.stop = slc.stop
          return itr
        def __iter__(self):
          for i in range(self.start, self.stop):
            chunk = self.chunklist[i]
            yield tasks.CopyTask(src_cv, dst_cv, src_z, dst_z, chunk, mip,
                                 is_field, to_uint8, masks)

    chunks = self.break_into_chunks(bbox, cm.dst_chunk_sizes[mip],
                                    cm.dst_voxel_offsets[mip], mip=mip,
                                    max_mip=cm.max_mip)
    if return_iterator:
        return CopyTaskIterator(chunks,0, len(chunks))
    #tq = GreenTaskQueue('deepalign_zhen')
    #tq.insert_all(ptasks, parallel=2)
    else:
        batch = []
        for chunk in chunks:
          batch.append(tasks.CopyTask(src_cv, dst_cv, src_z, dst_z, chunk, mip,
                                      is_field, to_uint8, masks))
        return batch

  def compute_field(self, cm, model_path, src_cv, tgt_cv, field_cv,
                    src_z, tgt_z, bbox, mip, pad=2048,
                    src_masks=[],
                    tgt_masks=[],
                    return_iterator=False, prev_field_cv=None, prev_field_z=None,
                    prev_field_inverse=False, coarse_field_cv=None,
                    coarse_field_mip=0,tgt_field_cv=None,stitch=False,report=False):
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
       prev_field_cv: MiplessCloudVolume where field prior is stored. Field will be used
        to apply initial translation to target image. If None, will ignore.
       prev_field_z: int for section index of previous field
       prev_field_inverse: bool indicating whether the inverse of the previous field
        should be used.

    """
    start = time()
    chunks = self.break_into_chunks(bbox, cm.dst_chunk_sizes[mip],
                                    cm.dst_voxel_offsets[mip], mip=mip,
                                    max_mip=cm.max_mip)
    class ComputeFieldTaskIterator():
        def __init__(self, cl, start, stop):
          self.chunklist = cl
          self.start = start
          self.stop = stop
        def __len__(self):
          return self.stop - self.start
        def __getitem__(self, slc):
          itr = deepcopy(self)
          itr.start = slc.start
          itr.stop = slc.stop
          return itr
        def __iter__(self):
          for i in range(self.start, self.stop):
            chunk = self.chunklist[i]
            yield tasks.ComputeFieldTask(model_path, src_cv, tgt_cv, field_cv,
                                          src_z, tgt_z, chunk, mip, pad,
                                          src_mask,
                                          tgt_mask,
                                          prev_field_cv, prev_field_z, prev_field_inverse,
                                          coarse_field_cv, coarse_field_mip, tgt_field_cv, stitch, report)
    if return_iterator:
        return ComputeFieldTaskIterator(chunks,0, len(chunks))
    else:
        batch = []
        for chunk in chunks:
          batch.append(tasks.ComputeFieldTask(model_path, src_cv, tgt_cv, field_cv,
                                              src_z, tgt_z, chunk, mip, pad,
                                              src_masks,
                                              tgt_masks,
                                              prev_field_cv, prev_field_z, prev_field_inverse,
                                              coarse_field_cv, coarse_field_mip, tgt_field_cv, stitch, report))
        return batch

  def seethrough_stitch_render(self, cm, src_cv, dst_cv, z_start, z_end,
                   bbox, mip, use_cpu=False, return_iterator=False, blackout_op='none'):
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
       affine: 2x3 ndarray for preconditioning affine to use (default: None means identity)
    """
    start = time()
    chunks = self.break_into_chunks(bbox, cm.dst_chunk_sizes[mip],
                                    cm.dst_voxel_offsets[mip], mip=mip,
                                    max_mip=cm.max_mip)
    class SeethroughStitchRenderTaskIterator():
        def __init__(self, cl, start, stop):
          self.chunklist = cl
          self.start = start
          self.stop = stop
        def __len__(self):
          return self.stop - self.start
        def __getitem__(self, slc):
          itr = deepcopy(self)
          itr.start = slc.start
          itr.stop = slc.stop
          return itr
        def __iter__(self):
          for i in range(self.start, self.stop):
            chunk = self.chunklist[i]
            yield tasks.SeethroughStitchRenderTask(src_cv,  dst_cv, z_start,
                       z_end, chunk, mip, use_cpu, blackout_op=blackout_op)
    if return_iterator:
        return SeethroughStitchRenderTaskIterator(chunks,0, len(chunks))
    else:
        batch = []
        for chunk in chunks:
          batch.append(tasks.SeethroughStitchRenderTask(src_cv,  dst_cv, z_start,
                       z_end, chunk, mip, use_cpu, blackout_op=blackout_op))

        return batch


  def render(self, cm, src_cv, field_cv, dst_cv, src_z, field_z, dst_z,
                   bbox, src_mip, field_mip,
                   masks=[],
                   affine=None, use_cpu=False,
             return_iterator= False, pad=256, seethrough=False,
             seethrough_misalign=False,
             blackout_op='none', report=False):
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
       affine: 2x3 ndarray for preconditioning affine to use (default: None means identity)
    """
    start = time()
    chunks = self.break_into_chunks(bbox, cm.dst_chunk_sizes[src_mip],
                                    cm.dst_voxel_offsets[src_mip], mip=src_mip,
                                    max_mip=cm.max_mip)
    class RenderTaskIterator():
        def __init__(self, cl, start, stop):
          self.chunklist = cl
          self.start = start
          self.stop = stop
        def __len__(self):
          return self.stop - self.start
        def __getitem__(self, slc):
          itr = deepcopy(self)
          itr.start = slc.start
          itr.stop = slc.stop
          return itr
        def __iter__(self):
          for i in range(self.start, self.stop):
            chunk = self.chunklist[i]
            yield tasks.RenderTask(src_cv, field_cv, dst_cv, src_z,
                       field_z, dst_z, chunk, src_mip, field_mip,
                       masks,
                       affine, use_cpu, pad,
                       seethrough=seethrough,
                       seethrough_misalign=seethrough_misalign,
                       blackout_op=blackout_op,
                       report=report)
    if return_iterator:
        return RenderTaskIterator(chunks,0, len(chunks))
    else:
        batch = []
        for chunk in chunks:
          batch.append(tasks.RenderTask(src_cv, field_cv, dst_cv, src_z,
                           field_z, dst_z, chunk, src_mip, field_mip,
                           masks,
                           affine, use_cpu, pad,
                           seethrough=seethrough,
                           seethrough_misalign=seethrough_misalign,
                           blackout_op=blackout_op,
                           report=report))
        return batch

  def vector_vote(self, cm, pairwise_cvs, vvote_cv, z, bbox, mip,
                  inverse=False, serial=True, return_iterator=False,
                  softmin_temp=None, blur_sigma=None):
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
       inverse: bool indicating if pairwise fields are to be treated as inverse fields
       serial: bool indicating to if a previously composed field is
        not necessary
       wait: bool indicating whether to wait for all tasks must finish before proceeding
    """
    start = time()
    chunks = self.break_into_chunks(bbox, cm.vec_chunk_sizes[mip],
                                    cm.vec_voxel_offsets[mip], mip=mip)
    class VvoteTaskIterator():
        def __init__(self, cl, start, stop):
          self.chunklist = cl
          self.start = start
          self.stop = stop
        def __len__(self):
          return self.stop - self.start
        def __getitem__(self, slc):
          itr = deepcopy(self)
          itr.start = slc.start
          itr.stop = slc.stop
          return itr
        def __iter__(self):
          for i in range(self.start, self.stop):
            chunk = self.chunklist[i]
            yield tasks.VectorVoteTask(deepcopy(pairwise_cvs), vvote_cv, z,
                                        chunk, mip, inverse, serial,
                                        softmin_temp=softmin_temp, blur_sigma=blur_sigma)
    if return_iterator:
        return VvoteTaskIterator(chunks,0, len(chunks))
    else:
        batch = []
        for chunk in chunks:
          batch.append(tasks.VectorVoteTask(deepcopy(pairwise_cvs), vvote_cv, z,
                                            chunk, mip, inverse, serial,
                                            softmin_temp=softmin_temp,
                                            blur_sigma=blur_sigma))
        return batch

  def compose(self, cm, f_cv, g_cv, dst_cv, f_z, g_z, dst_z, bbox,
                          f_mip, g_mip, dst_mip, factor, affine, pad,
                          return_iterator=False):
    """Compose two vector field CloudVolumes

    For coarse + fine composition:
      f = fine
      g = coarse

    Args:
       cm: CloudManager that corresponds to the f_cv, g_cv, dst_cv
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
       affine: affine matrix
       pad: padding size
    """
    start = time()
    chunks = self.break_into_chunks(bbox, cm.vec_chunk_sizes[dst_mip],
                                    cm.vec_voxel_offsets[dst_mip],
                                    mip=dst_mip)
    class CloudComposeIterator():
        def __init__(self, cl, start, stop):
          self.chunklist = cl
          self.start = start
          self.stop = stop
        def __len__(self):
          return self.stop - self.start
        def __getitem__(self, slc):
          itr = deepcopy(self)
          itr.start = slc.start
          itr.stop = slc.stop
          return itr
        def __iter__(self):
          for i in range(self.start, self.stop):
            chunk = self.chunklist[i]
            yield tasks.CloudComposeTask(f_cv, g_cv, dst_cv, f_z, g_z,
                                     dst_z, chunk, f_mip, g_mip, dst_mip,
                                     factor, affine, pad)
    if return_iterator:
        return CloudComposeIterator(chunks,0, len(chunks))
    else:
        batch = []
        for chunk in chunks:
          batch.append(tasks.CloudComposeTask(f_cv, g_cv, dst_cv, f_z, g_z,
                                         dst_z, chunk, f_mip, g_mip, dst_mip,
                                         factor, affine, pad))
        return batch

  def multi_compose(self, cm, cv_list, dst_cv, z_list, dst_z, bbox,
                                mip_list, dst_mip, factors, pad,
                                return_iterator=False):
    """Compose a list of field CloudVolumes

    This takes a list of fields
    field_list = [f_0, f_1, ..., f_n]
    and composes them to get
    f_0 ⚬ f_1 ⚬ ... ⚬ f_n ~= f_0(f_1(...(f_n)))

    Args:
       cm: CloudManager that corresponds to the f_cv, g_cv, dst_cv
       cv_list: list of MiplessCloudVolume storing the vector fields
       dst_cv: MiplessCloudVolume of composed vector field
       z_list: int or list of ints for section indices to read fields
       dst_z: int of section index to process
       bbox: BoundingBox of region to process
       mip_list: int or list of ints for MIPs of the input fields
       dst_mip: MIP of composed vector field
       pad: padding size
    """
    chunks = self.break_into_chunks(bbox, cm.vec_chunk_sizes[dst_mip],
                                    cm.vec_voxel_offsets[dst_mip],
                                    mip=dst_mip)
    if return_iterator:
        class CloudMultiComposeIterator():
            def __init__(self, cl, start, stop):
                self.chunklist = cl
                self.start = start
                self.stop = stop
            def __len__(self):
                return self.stop - self.start
            def __getitem__(self, slc):
                itr = deepcopy(self)
                itr.start = slc.start
                itr.stop = slc.stop
                return itr
            def __iter__(self):
                for i in range(self.start, self.stop):
                    chunk = self.chunklist[i]
                    yield tasks.CloudMultiComposeTask(cv_list, dst_cv, z_list,
                                                      dst_z, chunk, mip_list,
                                                      dst_mip, factors, pad)

        return CloudMultiComposeIterator(chunks, 0, len(chunks))
    else:
        batch = []
        for chunk in chunks:
            batch.append(tasks.CloudMultiComposeTask(cv_list, dst_cv, z_list,
                                                dst_z, chunk, mip_list,
                                                dst_mip, factors, pad))
        return batch

  def cloud_upsample_field(self, cm, src_cv, dst_cv, src_z, dst_z,
      bbox, src_mip, dst_mip, return_iterator=False):
    """Upsample Vector Field

    Args:
       cm: CloudManager that corresponds to the f_cv, g_cv, dst_cv
       src_cv: MiplessCloudVolume of (low res) source field
       dst_cv: MiplessCloudVolume of (higher res) destination field
       src_z: int of section index to process
       dst_z: int of section index to process
       bbox: BoundingBox of region to process
       src_mip: MIP of source vector field
       dst_mip: MIP of upsampled vector field
    """
    start = time()
    chunks = self.break_into_chunks(bbox, cm.vec_chunk_sizes[dst_mip],
                                    cm.vec_voxel_offsets[dst_mip],
                                    mip=dst_mip)
    class CloudUpsampleFieldIterator():
        def __init__(self, cl, start, stop):
          self.chunklist = cl
          self.start = start
          self.stop = stop
        def __len__(self):
          return self.stop - self.start
        def __getitem__(self, slc):
          itr = deepcopy(self)
          itr.start = slc.start
          itr.stop = slc.stop
          return itr
        def __iter__(self):
          for i in range(self.start, self.stop):
            chunk = self.chunklist[i]
            yield tasks.CloudUpsampleFieldTask(src_cv, dst_cv, src_z, dst_z,
                                     chunk, src_mip, dst_mip)
    if return_iterator:
        return CloudUpsampleFieldIterator(chunks,0, len(chunks))
    else:
        batch = []
        for chunk in chunks:
          batch.append(tasks.CloudUpsampleFieldTask(src_cv, dst_cv, src_z, dst_z,
                                         chunk, src_mip, dst_mip))
        return batch

  def cpc(self, cm, src_cv, tgt_cv, dst_cv, src_z, tgt_z, bbox, src_mip, dst_mip,
                norm=True, return_iterator=False):
    """Chunked Pearson Correlation between two CloudVolume images

    Args:
       cm: CloudManager that corresponds to the src_cv, tgt_cv, dst_cv
       src_cv: MiplessCloudVolume of source image
       tgt_cv: MiplessCloudVolume of target image
       dst_cv: MiplessCloudVolume of destination image
       src_z: int z index of one section to compare
       tgt_z: int z index of other section to compare
       bbox: BoundingBox of region to process
       src_mip: int MIP level of input src & tgt images
       dst_mip: int MIP level of output image, will dictate the size of the chunks
        used for the pearson r
       norm: bool for whether to normalize or not
    """
    start = time()
    chunks = self.break_into_chunks(bbox, cm.vec_chunk_sizes[dst_mip],
                                    cm.vec_voxel_offsets[dst_mip],
                                    mip=dst_mip)
    class CpcIterator():
        def __init__(self, cl, start, stop):
          self.chunklist = cl
          self.start = start
          self.stop = stop
        def __len__(self):
          return self.stop - self.start
        def __getitem__(self, slc):
          itr = deepcopy(self)
          itr.start = slc.start
          itr.stop = slc.stop
          return itr
        def __iter__(self):
          for i in range(self.start, self.stop):
            chunk = self.chunklist[i]
            yield tasks.CPCTask(src_cv, tgt_cv, dst_cv, src_z, tgt_z,
                                 chunk, src_mip, dst_mip, norm)
    if return_iterator:
        return CpcIterator(chunks,0, len(chunks))
    else:
        batch = []
        for chunk in chunks:
          batch.append(tasks.CPCTask(src_cv, tgt_cv, dst_cv, src_z, tgt_z,
                                     chunk, src_mip, dst_mip, norm))
        return batch

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

  def res_and_compose(self, model_path, src_cv, tgt_cv, z, tgt_range, bbox,
                      mip, write_F_cv, pad, softmin_temp):
      T = 2**mip
      fields = []
      for z_offset in tgt_range:
          src_z = z
          tgt_z = src_z - z_offset
          print("calc res for src {} and tgt {}".format(src_z, tgt_z))
          f = self.compute_field_chunk(model_path, src_cv, tgt_cv, src_z,
                                       tgt_z, bbox, mip, pad)
          #print("--------f shape is ---", f.shape)
          fields.append(f)
          #fields.append(f)
      fields = [torch.from_numpy(i).to(device=self.device) for i in fields]
      #print("device is ", fields[0].device)
      field = vector_vote(fields, softmin_temp=softmin_temp)
      field = field.data.cpu().numpy()
      self.save_field(field, write_F_cv, z, bbox, mip, relative=False)

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

  def compute_field_and_vector_vote(self, cm, model_path, src_cv, tgt_cv, vvote_field,
                          tgt_range, z, bbox, mip, pad, softmin_temp):
    """Create all pairwise matches for each SRC_Z in Z_RANGE to each TGT_Z in
    TGT_RADIUS and perform vetor voting
    """

    m = mip
    chunks = self.break_into_chunks(bbox, cm.vec_chunk_sizes[m],
                                    cm.vec_voxel_offsets[m], mip=m,
                                    max_mip=cm.num_scales)
    batch = []
    for patch_bbox in chunks:
        batch.append(tasks.ResAndComposeTask(model_path, src_cv, tgt_cv, z,
                                            tgt_range, patch_bbox, mip,
                                            vvote_field, pad, softmin_temp))
    return batch

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

  def rechunck_image(self, chunk_size, image):
      I = image.split(chunk_size, dim=2)
      I = torch.cat(I, dim=0)
      I = I.split(chunk_size, dim=3)
      return torch.cat(I, dim=1)

  def sum_pool(self, cm, src_cv, dst_cv, src_z, dst_z, bbox, src_mip, dst_mip):
      chunks = self.break_into_chunks(bbox, self.chunk_size,
                                      cm.dst_voxel_offsets[dst_mip], mip=src_mip,
                                      max_mip=cm.max_mip)
      batch = []
      for chunk in chunks:
        batch.append(tasks.SumPoolTask(src_cv, dst_cv, src_z,
                                     dst_z, chunk, src_mip, dst_mip))
      return batch

  def dilation(self, cm, src_cv, dst_cv, src_z, dst_z, bbox, mip, radius=3):
      chunks = self.break_into_chunks(bbox, self.chunk_size,
                                      cm.dst_voxel_offsets[mip], mip=mip,
                                      max_mip=cm.max_mip)
      batch = []
      for chunk in chunks:
        batch.append(tasks.Dilation(src_cv, dst_cv, src_z, dst_z, chunk, mip,
                                    radius))
      return batch

  def threshold(self, cm, src_cv, dst_cv, src_z, dst_z, bbox, mip, threshold=0, op='<'):
      chunks = self.break_into_chunks(bbox, self.chunk_size,
                                      cm.dst_voxel_offsets[mip], mip=mip,
                                      max_mip=cm.max_mip)
      batch = []
      for chunk in chunks:
        batch.append(tasks.Threshold(src_cv, dst_cv, src_z, dst_z, chunk, mip,
                                     threshold, op))
      return batch

  def compute_smoothness(self, cm, src_cv, dst_cv, src_z, dst_z, bbox, mip):
      chunks = self.break_into_chunks(bbox, self.chunk_size,
                                      cm.dst_voxel_offsets[mip], mip=mip,
                                      max_mip=cm.max_mip)
      batch = []
      for chunk in chunks:
        batch.append(tasks.ComputeSmoothness(src_cv, dst_cv, src_z, dst_z, chunk, mip))
      return batch

  def compute_smoothness_chunk(self, cv, z, bbox, mip, pad):
      padded_bbox = deepcopy(bbox)
      padded_bbox.max_mip = mip
      padded_bbox.uncrop(pad, mip=mip)
      field = self.get_field(cv, z, padded_bbox, mip, relative=False, to_tensor=True)
      return lap([field], device=self.device).unsqueeze(0)

  def compute_fcorr(self, cm, src_cv, dst_pre_cv, dst_post_cv, bbox, src_mip,
                    dst_mip, src_z, tgt_z, dst_z, fcorr_chunk_size, fill_value=0):
      chunks = self.break_into_chunks(bbox, self.chunk_size,
                                      cm.dst_voxel_offsets[dst_mip], mip=dst_mip,
                                      max_mip=cm.max_mip)
      batch = []
      for chunk in chunks:
        batch.append(tasks.ComputeFcorrTask(src_cv, dst_pre_cv, dst_post_cv, chunk,
                                            src_mip, dst_mip, src_z, tgt_z, dst_z,
                                            fcorr_chunk_size, fill_value))
      return batch

  def get_fcorr(self, cv, src_z, tgt_z, bbox, mip, chunk_size=16, fill_value=0):
      """Perform fcorr for two images
      """
      src = self.get_data(cv, src_z, bbox, src_mip=mip, dst_mip=mip,
                             to_float=False, to_tensor=True).float()
      tgt = self.get_data(cv, tgt_z, bbox, src_mip=mip, dst_mip=mip,
                             to_float=False, to_tensor=True).float()

      std1 = src[src!=0].std()
      std2 = tgt[tgt!=0].std()
      scaling = 8 * pow(std1*std2, 1/2)
      # scaling = 240 # Fixed threshold
      # scaling = 2000

      new_image1 = self.rechunck_image(chunk_size, src)
      new_image2 = self.rechunck_image(chunk_size, tgt)
      f1, p1 = get_fft_power2(new_image1)
      f2, p2 = get_fft_power2(new_image2)
      tmp_image = get_hp_fcorr(f1, p1, f2, p2, scaling=scaling, fill_value=fill_value)
      tmp_image = tmp_image.permute(2,3,0,1)
      tmp_image = tmp_image.cpu().numpy()
      tmp = deepcopy(tmp_image)
      tmp[tmp==2]=1
      std = 1.
      blurred = scipy.ndimage.morphology.filters.gaussian_filter(tmp, sigma=(0, 0, std, std))
      s = scipy.ndimage.generate_binary_structure(2, 1)[None, None, :, :]
      closed = scipy.ndimage.morphology.grey_closing(blurred, footprint=s)
      closed = 2*closed
      closed[closed>1] = 1
      closed = 1-closed
      #print("++++closed shape",closed.shape)
      return closed, tmp_image

  def get_ones(self, bbox, mip):
      x_range = bbox.x_range(mip=mip)
      y_range = bbox.y_range(mip=mip)
      return np.ones([x_range[1]-x_range[0], y_range[1]-y_range[0]])

  def mask_conjunction_chunk(self, cv_list, z_list, bbox, mip_list, dst_mip):
      mask = self.get_data(cv_list[0], z_list[0], bbox, src_mip=mip_list[0],
                           dst_mip=dst_mip, to_float=False, to_tensor=False)
      for cv, z, mip in zip(cv_list[1:], z_list[1:], mip_list[1:]):
        mask = np.logical_and(mask, self.get_data(cv, z, bbox, src_mip=mip,
                                                  dst_mip=dst_mip, to_float=False,
                                                  to_tensor=False))
      return mask

  def mask_disjunction_chunk(self, cv_list, z_list, bbox, mip_list, dst_mip):
      mask = self.get_data(cv_list[0], z_list[0], bbox, src_mip=mip_list[0],
                           dst_mip=dst_mip, to_float=False, to_tensor=False)
      for cv, z, mip in zip(cv_list[1:], z_list[1:], mip_list[1:]):
        mask = np.logical_or(mask, self.get_data(cv, z, bbox, src_mip=mip,
                                                  dst_mip=dst_mip, to_float=False,
                                                  to_tensor=False))
      return mask

  def filterthree_op_chunk(self, bbox, mask_cv, z, mip):
      mask1 = self.get_data(mask_cv, z, bbox, src_mip=mip, dst_mip=mip,
                                to_float=False, to_tensor=False)
      mask2 = self.get_data(mask_cv, z+1, bbox, src_mip=mip, dst_mip=mip,
                                to_float=False, to_tensor=False)
      mask3 = self.get_data(mask_cv, z+2, bbox, src_mip=mip, dst_mip=mip,
                                to_float=False, to_tensor=False)

      return np.logical_and(np.logical_and(mask1, mask2), mask3)

  def filterthree_op(self, cm, bbox, mask_cv, dst_cv, z, dst_z, mip):
      chunks = self.break_into_chunks(bbox, cm.dst_chunk_sizes[mip],
                                      cm.dst_voxel_offsets[mip],
                                      mip=mip, max_mip=cm.max_mip)
      batch = []
      for chunk in chunks:
        batch.append(tasks.FilterThreeOpTask(chunk, mask_cv, dst_cv, z, dst_z, mip))
      return batch


  def make_fcorr_masks(self, cm, cv_list, dst_pre, dst_post, z_list, dst_z,
                       bbox, mip, operators, threshold, dilate_radius=0):
      chunks = self.break_into_chunks(bbox, cm.dst_chunk_sizes[mip],
                                      cm.dst_voxel_offsets[mip], mip=mip,
                                      max_mip=cm.max_mip)
      batch = []
      for chunk in chunks:
        batch.append(tasks.FcorrMaskTask(cv_list, dst_pre, dst_post, z_list, dst_z,
                                         chunk, mip, operators, threshold, dilate_radius))
      return batch

  def mask_logic(self, cm, cv_list, dst_cv, z_list, dst_z, bbox, mip_list,
                 dst_mip, op='or'):
      chunks = self.break_into_chunks(bbox, cm.dst_chunk_sizes[dst_mip],
                                      cm.dst_voxel_offsets[dst_mip],
                                      mip=dst_mip, max_mip=cm.max_mip)
      batch = []
      for chunk in chunks:
        batch.append(tasks.MaskLogicTask(cv_list, dst_cv, z_list, dst_z, chunk,
                                         mip_list, dst_mip, op))
      return batch

  def mask_section(self, cm, bbox, cv, z, mip):
      chunks = self.break_into_chunks(bbox, cm.dst_chunk_sizes[mip],
                                      cm.vec_voxel_offsets[mip],
                                      mip=mip, max_mip=cm.max_mip)
      batch = []
      for chunk in chunks:
        batch.append(tasks.MaskOutTask(cv, mip, z, chunk))
      return batch

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
          # sleep(1.75)
          sleep(5)
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
  @retry
  def sqs_is_empty(self):
    # hashtag hackerlife
    attribute_names = ['ApproximateNumberOfMessages', 'ApproximateNumberOfMessagesNotVisible']
    responses = []
    for i in range(3):
      response = self.sqs.get_queue_attributes(QueueUrl=self.queue_url,
                                               AttributeNames=attribute_names)
      for a in attribute_names:
        responses.append(int(response['Attributes'][a]))
      print('{}     '.format(responses[-2:]), end="\r", flush=True)
      if i < 2:
        sleep(2)
    return all(i == 0 for i in responses)

  def wait_for_sqs_empty(self):
    self.sqs = boto3.client('sqs', region_name='us-east-1')
    self.queue_url  = self.sqs.get_queue_url(QueueName=self.queue_name)["QueueUrl"]
    print("\nSQS Wait")
    print("No. of messages / No. not visible")
    sleep(5)
    while not self.sqs_is_empty():
      sleep(1)
