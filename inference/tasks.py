import boto3
from time import time
import torch
from torch.nn.functional import conv2d
import json
import tenacity
import operator
import numpy as np
from copy import deepcopy
from os.path import join
from functools import partial
from mipless_cloudvolume import deserialize_miplessCV as DCV
from cloudvolume import Storage
from cloudvolume.lib import scatter 
from boundingbox import BoundingBox, deserialize_bbox
from fcorr import fcorr_conjunction
from scipy import ndimage

from taskqueue import RegisteredTask, TaskQueue, LocalTaskQueue, GreenTaskQueue
from concurrent.futures import ProcessPoolExecutor
# from taskqueue.taskqueue import _scatter as scatter

def remote_upload(queue_name, ptasks):
  with TaskQueue(queue_name=queue_name) as tq:
    for task in ptasks:
      tq.insert(task)

def green_upload(ptask, aligner):
    if aligner.distributed:
        tq = GreenTaskQueue(aligner.queue_name)
        tq.insert_all(ptask, parallel=aligner.threads)
    else:
        tq = LocalTaskQueue(parallel=1)
        tq.insert_all(ptask, args= [aligner])
   
   # for task in ptask:
   #     tq.insert(task, args=[ a ])


def run(aligner, tasks): 
  if aligner.distributed:
    tasks = scatter(tasks, aligner.threads)
    fn = partial(remote_upload, aligner.queue_name)
    with ProcessPoolExecutor(max_workers=aligner.threads) as executor:
      executor.map(fn, tasks)
  else:
    with LocalTaskQueue(queue_name=aligner.queue_name, parallel=1) as tq:
      for task in tasks:
        tq.insert(task, args=[ aligner ])

class PredictImageTask(RegisteredTask):
  def __init__(self, model_path, src_cv, dst_cv, z, mip, bbox):
    super().__init__(model_path, src_cv, dst_cv, z, mip, bbox)

  def execute(self, aligner):
    src_cv = DCV(self.src_cv)
    dst_cv = DCV(self.dst_cv)
    z = self.z
    patch_bbox = deserialize_bbox(self.bbox)
    mip = self.mip

    print("\nPredict Image\n"
          "src {}\n"
          "dst {}\n"
          "at z={}\n"
          "MIP{}\n".format(src_cv, dst_cv, z, mip), flush=True)
    start = time()
    image = aligner.predict_image_chunk(self.model_path, src_cv, z, mip, patch_bbox)
    image = image.cpu().numpy()
    aligner.save_image(image, dst_cv, z, patch_bbox, mip)

    end = time()
    diff = end - start
    print(':{:.3f} s'.format(diff))


class CopyTask(RegisteredTask):
  def __init__(self, src_cv, dst_cv, src_z, dst_z, patch_bbox, mip, 
               is_field, to_uint8, mask_cv, mask_mip, mask_val):
    super().__init__(src_cv, dst_cv, src_z, dst_z, patch_bbox, mip, 
                     is_field, to_uint8, mask_cv, mask_mip, mask_val)

  def execute(self, aligner):
    src_cv = DCV(self.src_cv)
    dst_cv = DCV(self.dst_cv)
    src_z = self.src_z
    dst_z = self.dst_z
    patch_bbox = deserialize_bbox(self.patch_bbox)
    mip = self.mip
    is_field = self.is_field
    to_uint8 = self.to_uint8
    mask_cv = None 
    if self.mask_cv:
      mask_cv = DCV(self.mask_cv)
    mask_mip = self.mask_mip
    mask_val = self.mask_val

    print("\nCopy\n"
          "src {}\n"
          "dst {}\n"
          "mask {}, val {}, MIP{}\n"
          "z={} to z={}\n"
          "MIP{}\n".format(src_cv, dst_cv, mask_cv, mask_val, mask_mip, 
                            src_z, dst_z, mip), flush=True)
    start = time()
    if not aligner.dry_run:
      if is_field:
        field =  aligner.get_field(src_cv, src_z, patch_bbox, mip, relative=False,
                                to_tensor=False)
        aligner.save_field(field, dst_cv, dst_z, patch_bbox, mip, relative=False)
      elif to_uint8:
        image = aligner.get_masked_image(src_cv, src_z, patch_bbox, mip,
                                mask_cv=mask_cv, mask_mip=mask_mip,
                                mask_val=mask_val,
                                to_tensor=False, normalizer=None)
        aligner.save_image(image, dst_cv, dst_z, patch_bbox, mip, to_uint8=True)
      else:
        image = aligner.get_data(src_cv, src_z, patch_bbox, mip, mip, to_float=False,
                                 to_tensor=False, normalizer=None)
        aligner.save_image(image, dst_cv, dst_z, patch_bbox, mip, to_uint8=False)
      end = time()
      diff = end - start
      print(':{:.3f} s'.format(diff))

class ComputeFieldTask(RegisteredTask):
  def __init__(self, model_path, src_cv, tgt_cv, field_cv, src_z, tgt_z, 
                     patch_bbox, mip, pad, src_mask_cv, src_mask_val, src_mask_mip, 
                     tgt_mask_cv, tgt_mask_val, tgt_mask_mip,
                     prev_field_cv, prev_field_z, prev_field_inverse):
    super().__init__(model_path, src_cv, tgt_cv, field_cv, src_z, tgt_z, 
                     patch_bbox, mip, pad, src_mask_cv, src_mask_val, src_mask_mip, 
                     tgt_mask_cv, tgt_mask_val, tgt_mask_mip,
                     prev_field_cv, prev_field_z, prev_field_inverse)

  def execute(self, aligner):
    model_path = self.model_path
    src_cv = DCV(self.src_cv) 
    tgt_cv = DCV(self.tgt_cv) 
    field_cv = DCV(self.field_cv)
    if self.prev_field_cv is not None:
        prev_field_cv = DCV(self.prev_field_cv)
    else:
        prev_field_cv = None
    src_z = self.src_z
    tgt_z = self.tgt_z
    prev_field_z = self.prev_field_z
    prev_field_inverse = self.prev_field_inverse
    patch_bbox = deserialize_bbox(self.patch_bbox)
    mip = self.mip
    pad = self.pad
    src_mask_cv = None 
    if self.src_mask_cv:
      src_mask_cv = DCV(self.src_mask_cv)
    src_mask_mip = self.src_mask_mip
    src_mask_val = self.src_mask_val
    tgt_mask_cv = None 
    if self.tgt_mask_cv:
      tgt_mask_cv = DCV(self.tgt_mask_cv)
    tgt_mask_mip = self.tgt_mask_mip
    tgt_mask_val = self.tgt_mask_val

    print("\nCompute field\n"
          "model {}\n"
          "src {}\n"
          "tgt {}\n"
          "field {}\n"
          "src_mask {}, val {}, MIP{}\n"
          "tgt_mask {}, val {}, MIP{}\n"
          "z={} to z={}\n"
          "MIP{}\n".format(model_path, src_cv, tgt_cv, field_cv, src_mask_cv, src_mask_val,
                           src_mask_mip, tgt_mask_cv, tgt_mask_val, tgt_mask_mip, 
                           src_z, tgt_z, mip), flush=True)
    start = time()
    if not aligner.dry_run:
      field = aligner.compute_field_chunk(model_path, src_cv, tgt_cv, src_z, tgt_z, 
                                          patch_bbox, mip, pad, 
                                          src_mask_cv, src_mask_mip, src_mask_val,
                                          tgt_mask_cv, tgt_mask_mip, tgt_mask_val,
                                          None, prev_field_cv, prev_field_z, 
                                          prev_field_inverse)
      aligner.save_field(field, field_cv, src_z, patch_bbox, mip, relative=False)
      end = time()
      diff = end - start
      print('ComputeFieldTask: {:.3f} s'.format(diff))

class RenderTask(RegisteredTask):
  def __init__(self, src_cv, field_cv, dst_cv, src_z, field_z, dst_z, patch_bbox, src_mip,
               field_mip, mask_cv, mask_mip, mask_val, affine, use_cpu=False):
    super(). __init__(src_cv, field_cv, dst_cv, src_z, field_z, dst_z, patch_bbox, src_mip, 
                     field_mip, mask_cv, mask_mip, mask_val, affine, use_cpu)

  def execute(self, aligner):
    src_cv = DCV(self.src_cv) 
    field_cv = DCV(self.field_cv) 
    dst_cv = DCV(self.dst_cv) 
    src_z = self.src_z
    field_z = self.field_z
    dst_z = self.dst_z
    patch_bbox = deserialize_bbox(self.patch_bbox)
    src_mip = self.src_mip
    field_mip = self.field_mip
    mask_cv = None 
    if self.mask_cv:
      mask_cv = DCV(self.mask_cv)
    mask_mip = self.mask_mip
    mask_val = self.mask_val
    affine = None 
    if self.affine:
      affine = np.array(self.affine)

    print("\nRendering\n"
          "src {}\n"
          "field {}\n"
          "dst {}\n"
          "z={} to z={}\n"
          "MIP{} to MIP{}\n"
          "\n".format(src_cv.path, field_cv.path, dst_cv.path, src_z, dst_z, 
                        field_mip, src_mip), flush=True)
    start = time()
    if not aligner.dry_run:
      image = aligner.cloudsample_image(src_cv, field_cv, src_z, field_z,
                                     patch_bbox, src_mip, field_mip,
                                     mask_cv=mask_cv, mask_mip=mask_mip,
                                     mask_val=mask_val, affine=affine,
                                     use_cpu=self.use_cpu)
      image = image.cpu().numpy()
      aligner.save_image(image, dst_cv, dst_z, patch_bbox, src_mip)
      end = time()
      diff = end - start
      print('RenderTask: {:.3f} s'.format(diff))

class VectorVoteTask(RegisteredTask):
  def __init__(self, pairwise_cvs, vvote_cv, z, patch_bbox, mip, inverse, serial,
               softmin_temp, blur_sigma):
    super().__init__(pairwise_cvs, vvote_cv, z, patch_bbox, mip, inverse, serial,
                     softmin_temp, blur_sigma)

  def execute(self, aligner):
    pairwise_cvs = {int(k): DCV(v) for k,v in self.pairwise_cvs.items()}
    vvote_cv = DCV(self.vvote_cv)
    z = self.z
    patch_bbox = deserialize_bbox(self.patch_bbox)
    mip = self.mip
    inverse = bool(self.inverse)
    serial = bool(self.serial)

    softmin_temp = self.softmin_temp
    blur_sigma = self.blur_sigma
    print("\nVector vote\n"
          "fields {}\n"
          "dst {}\n"
          "z={}\n"
          "MIP{}\n"
          "inverse={}\n"
          "serial={}\n"
          "softmin_temp={}\n"
          "blur_sigma={}\n".format(pairwise_cvs.keys(), vvote_cv, z, 
                                   mip, inverse, serial, softmin_temp,
                                   blur_sigma), flush=True)
    start = time()
    if not aligner.dry_run:
      field = aligner.vector_vote_chunk(pairwise_cvs, vvote_cv, z, patch_bbox, mip, 
                                        inverse=inverse, serial=serial, 
                                        softmin_temp=softmin_temp, blur_sigma=blur_sigma)
      field = field.data.cpu().numpy()
      aligner.save_field(field, vvote_cv, z, patch_bbox, mip, relative=False)
      end = time()
      diff = end - start
      print('VectorVoteTask: {:.3f} s'.format(diff))


class CloudComposeTask(RegisteredTask):
  def __init__(self, f_cv, g_cv, dst_cv, f_z, g_z, dst_z, patch_bbox, f_mip, g_mip, 
                     dst_mip, factor, affine, pad):
    super().__init__(f_cv, g_cv, dst_cv, f_z, g_z, dst_z, patch_bbox, f_mip, g_mip, 
                     dst_mip, factor, affine, pad)

  def execute(self, aligner):
    f_cv = DCV(self.f_cv)
    g_cv = DCV(self.g_cv)
    dst_cv = DCV(self.dst_cv)
    f_z = self.f_z
    g_z = self.g_z
    dst_z = self.dst_z
    patch_bbox = deserialize_bbox(self.patch_bbox)
    f_mip = self.f_mip
    g_mip = self.g_mip
    dst_mip = self.dst_mip
    factor = self.factor
    pad = self.pad
    affine = None
    if self.affine:
      affine = np.array(self.affine)

    print("\nCompose\n"
          "f {}\n"
          "g {}\n"
          "f_z={}, g_z={}\n"
          "f_MIP{}, g_MIP{}\n"
          "dst {}\n"
          "dst_MIP {}\n".format(f_cv, g_cv, f_z, g_z, f_mip, g_mip, dst_cv, 
                               dst_mip), flush=True)
    start = time()
    if not aligner.dry_run:
      h = aligner.cloudsample_compose(f_cv, g_cv, f_z, g_z, patch_bbox, f_mip,
                                     g_mip, dst_mip, factor=factor,
                                     affine=affine, pad=pad)
      h = h.data.cpu().numpy()
      aligner.save_field(h, dst_cv, dst_z, patch_bbox, dst_mip, relative=False)
      end = time()
      diff = end - start
      print('ComposeTask: {:.3f} s'.format(diff))


class CloudMultiComposeTask(RegisteredTask):
    def __init__(self, cv_list, dst_cv, z_list, dst_z, patch_bbox, mip_list,
                 dst_mip, factors, pad):
        super().__init__(cv_list, dst_cv, z_list, dst_z, patch_bbox, mip_list,
                         dst_mip, factors, pad)

    def execute(self, aligner):
        cv_list = [DCV(f) for f in self.cv_list]
        dst_cv = DCV(self.dst_cv)
        z_list = self.z_list
        dst_z = self.dst_z
        patch_bbox = deserialize_bbox(self.patch_bbox)
        mip_list = self.mip_list
        dst_mip = self.dst_mip
        factors = self.factors
        pad = self.pad
    
        print("\nCompose\n"
              "cv {}\n"
              "z={}\n"
              "MIPs={}\n"
              "dst {}\n"
              "dst_MIP {}\n"
              .format(cv_list, z_list, mip_list, dst_cv, dst_mip),
              flush=True)
        start = time()
        if not aligner.dry_run:
            h = aligner.cloudsample_multi_compose(cv_list, z_list, patch_bbox,
                                                  mip_list, dst_mip, factors,
                                                  pad)
            h = h.data.cpu().numpy()
            aligner.save_field(h, dst_cv, dst_z, patch_bbox, dst_mip,
                               relative=False)
            end = time()
            diff = end - start
            print('MultiComposeTask: {:.3f} s'.format(diff))


class CPCTask(RegisteredTask):
  def __init__(self, src_cv, tgt_cv, dst_cv, src_z, tgt_z, patch_bbox, 
                    src_mip, dst_mip, norm):
    super().__init__(src_cv, tgt_cv, dst_cv, src_z, tgt_z, patch_bbox, 
                    src_mip, dst_mip, norm)

  def execute(self, aligner):
    src_cv = DCV(self.src_cv) 
    tgt_cv = DCV(self.tgt_cv) 
    dst_cv = DCV(self.dst_cv)
    src_z = self.src_z
    tgt_z = self.tgt_z
    patch_bbox = deserialize_bbox(self.patch_bbox)
    src_mip = self.src_mip
    dst_mip = self.dst_mip
    norm = self.norm
    print("\nCPC\n"
          "src {}\n"
          "tgt {}\n"
          "src_z={}, tgt_z={}\n"
          "src_MIP{} to dst_MIP{}\n"
          "norm={}\n"
          "dst {}\n".format(src_cv, tgt_cv, src_z, tgt_z, src_mip, dst_mip, norm,
                            dst_cv), flush=True)
    if not aligner.dry_run:
      r = aligner.cpc_chunk(src_cv, tgt_cv, src_z, tgt_z, patch_bbox, src_mip, 
                            dst_mip, norm)
      r = r.cpu().numpy()
      aligner.save_image(r, dst_cv, src_z, patch_bbox, dst_mip, to_uint8=norm)

class BatchRenderTask(RegisteredTask):
  def __init__(
    self, z, field_cv, field_z, patches, 
    mip, dst_cv, dst_z, batch
  ):
    super().__init__(
      z, field_cv, field_z, patches, 
      mip, dst_cv, dst_z, batch
    )
    #self.patches = [p.serialize() for p in patches]

  def execute(self, aligner):
    src_z = self.z
    patches  = [deserialize_bbox(p) for p in self.patches]
    batch = self.batch
    field_cv = DCV(self.field_cv)
    mip = self.mip
    field_z = self.field_z
    dst_cv = DCV(self.dst_cv)
    dst_z = self.dst_z

    def chunkwise(patch_bbox):
      print ("Rendering {} at mip {}".format(patch_bbox.__str__(mip=0), mip),
              end='', flush=True)
      warped_patch = aligner.warp_patch_batch(src_z, field_cv, field_z,
                                           patch_bbox, mip, batch)
      aligner.save_image_patch_batch(dst_cv, (dst_z, dst_z + batch),
                                  warped_patch, patch_bbox, mip)
    aligner.pool.map(chunkwise, patches)

class DownsampleTask(RegisteredTask):
  def __init__(self, cv, z, patches, mip):
    super().__init__(cv, z, patches, mip)
    #self.patches = [p.serialize() for p in patches]

  def execute(self, aligner):
    z = self.z
    cv = DCV(self.cv)
    #patches  = deserialize_bbox(self.patches)
    patches  = [deserialize_bbox(p) for p in self.patches]
    mip = self.mip
    #downsampled_patch = aligner.downsample_patch(cv, z, patches, mip - 1)
    #aligner.save_image_patch(cv, z, downsampled_patch, patches, mip)
    def chunkwise(patch_bbox):
      downsampled_patch = aligner.downsample_patch(cv, z, patch_bbox, mip - 1)
      aligner.save_image_patch(cv, z, downsampled_patch, patch_bbox, mip)
    aligner.pool.map(chunkwise, patches)

class InvertFieldTask(RegisteredTask):
  def __init__(self, z, src_cv, dst_cv, patch_bbox, mip, optimizer):
    super().__init__(z, src_cv, dst_cv, patch_bbox, mip, optimizer)

  def execute(self, aligner):
    src_cv = DCV(self.src_cv)
    dst_cv = DCV(self.dst_cv)
    patch_bbox = deserialize_bbox(self.patch_bbox)

    aligner.invert_field(
      self.z, src_cv, dst_cv,
      patch_bbox, self.mip, self.optimizer
    )

class PrepareTask(RegisteredTask):
  def __init__(self, z, patches, mip, start_z):
    super().__init__(z, patches, mip, start_z)
    #self.patches = [ p.serialize() for p in patches ]

  def execute(self, aligner):
    patches = [ deserialize_bbox(p) for p in self.patches ]

    def chunkwise(patch_bbox):
      print("Preparing source {} at mip {}".format(
        patch_bbox.__str__(mip=0), mip
      ), end='', flush=True)

      warped_patch = aligner.warp_patch(
        aligner.src_ng_path, self.z, patch_bbox,
        (self.mip, aligner.process_high_mip), 
        self.mip, self.start_z
      )
      aligner.save_image_patch(
        aligner.tmp_ng_path, warped_patch, self.z, patch_bbox, self.mip
      )

    aligner.pool.map(chunkwise, patches)    

class RegularizeTask(RegisteredTask):
  def __init__(self, z_start, z_end, compose_start, patch_bbox, mip, sigma):
    super().__init(z_start, z_end, compose_start, patch_bbox, mip, sigma)

  def execute(self, aligner):
    patch_bbox = deserialize_bbox(self.patch_bbox)
    z_range = range(self.z_start, self.z_end+1)
    
    aligner.regularize_z(
      z_range, self.compose_start, 
      patch_bbox, self.mip, 
      sigma=self.sigma
    )    

class RenderCVTask(RegisteredTask):
  def __init__(self, z, field_cv, field_z, patches, mip, dst_cv, dst_z):
    super().__init__(z, field_cv, field_z, patches, mip, dst_cv, dst_z)
    #self.patches = [p.serialize() for p in patches]

  def execute(self, aligner):
    src_z = self.z
    patches  = [deserialize_bbox(p) for p in self.patches]
    #patches  = deserialize_bbox(self.patches)
    field_cv = DCV(self.field_cv) 
    mip = self.mip
    field_z = self.field_z
    dst_cv = DCV(self.dst_cv)
    dst_z = self.dst_z

    def chunkwise(patch_bbox):
      print ("Rendering {} at mip {}".format(patch_bbox.__str__(mip=0), mip),
              end='', flush=True)
      warped_patch = aligner.warp_using_gridsample_cv(src_z, field_cv, field_z, patch_bbox, mip)
      aligner.save_image_patch(dst_cv, dst_z, warped_patch, patch_bbox, mip)
    aligner.pool.map(chunkwise, patches)    

class RenderLowMipTask(RegisteredTask):
  def __init__(
    self, z, field_cv, field_z, patches, 
    image_mip, vector_mip, dst_cv, dst_z
  ):
    super().__init__(
      z, field_cv, field_z, patches, 
      image_mip, vector_mip, dst_cv, dst_z
    )
    #self.patches = [p.serialize() for p in patches]

  def execute(self, aligner):
    src_z = self.z
    patches  = [deserialize_bbox(p) for p in self.patches]
    field_cv = DCV(self.field_cv) 
    image_mip = self.image_mip
    vector_mip = self.vector_mip
    field_z = self.field_z
    dst_cv = DCV(self.dst_cv)
    dst_z = self.dst_z
    def chunkwise(patch_bbox):
      print ("Rendering {} at mip {}".format(patch_bbox.__str__(mip=0), image_mip),
              end='', flush=True)
      warped_patch = aligner.warp_patch_at_low_mip(src_z, field_cv, field_z, 
                                                patch_bbox, image_mip, vector_mip)
      aligner.save_image_patch(dst_cv, dst_z, warped_patch, patch_bbox, image_mip)
    aligner.pool.map(chunkwise, patches)

class ResAndComposeTask(RegisteredTask):
  def __init__(self, model_path, src_cv, tgt_cv, z, tgt_range, patch_bbox, mip,
               w_cv, pad, softmin_temp):
    super().__init__(model_path, src_cv, tgt_cv, z, tgt_range, patch_bbox, mip,
               w_cv, pad, softmin_temp)

  def execute(self, aligner):
    patch_bbox = deserialize_bbox(self.patch_bbox)
    w_cv = DCV(self.w_cv)
    src_cv = DCV(self.src_cv)
    tgt_cv = DCV(self.tgt_cv)
    print("self tgt_range is", self.tgt_range)
    aligner.res_and_compose(self.model_path, src_cv, tgt_cv, self.z,
                            self.tgt_range, patch_bbox, self.mip, w_cv,
                            self.pad, self.softmin_temp)

class UpsampleRenderRechunkTask(RegisteredTask):
  def __init__(
    self, z_range, src_cv, field_cv, dst_cv, 
    patches, image_mip, field_mip
  ):
    super().__init__(
      z_range, src_cv, field_cv, dst_cv, 
      patches, image_mip, field_mip
    )
    #self.patches = [p.serialize() for p in patches]

  def execute(self, aligner):
    z_start = self.z_start
    z_end = self.z_end
    patches  = [deserialize_bbox(p) for p in self.patches]
    #patches  = deserialize_bbox(self.patches)
    src_cv = DCV(self.src_cv) 
    field_cv = DCV(self.field_cv) 
    dst_cv = DCV(self.dst_cv)
    image_mip = self.image_mip
    field_mip = self.field_mip
    z_range = range(z_start, z_end+1)
    def chunkwise(patch_bbox):
      warped_patch = aligner.warp_gridsample_cv_batch(z_range, src_cv, field_cv, 
                                                   patch_bbox, image_mip, field_mip)
      print('warped_patch.shape {0}'.format(warped_patch.shape))
      aligner.save_image_patch_batch(dst_cv, (z_range[0], z_range[-1]+1), warped_patch, 
                                  patch_bbox, image_mip)
    aligner.pool.map(chunkwise, patches)

class FilterThreeOpTask(RegisteredTask):
  def __init__(self, bbox, mask_cv, dst_cv, z, dst_z, mip):
    super(). __init__(bbox, mask_cv, dst_cv, z, dst_z, mip)

  def execute(self, aligner):
    mask_cv = DCV(self.mask_cv)
    dst_cv = DCV(self.dst_cv)
    z = self.z
    dst_z = self.dst_z
    patch_bbox = deserialize_bbox(self.bbox)
    mip = self.mip
    print("\n Mask conjunction \n" )
    start = time()
    res = aligner.filterthree_op_chunk(patch_bbox, mask_cv, z, mip)
    aligner.append_image(res, dst_cv, dst_z, patch_bbox, mip, to_uint8=True)
    aligner.append_image(res, dst_cv, dst_z+1, patch_bbox, mip, to_uint8=True)
    aligner.append_image(res, dst_cv, dst_z+2, patch_bbox, mip, to_uint8=True)
    end = time()
    diff = end - start
    print('Task: {:.3f} s'.format(diff))

class FcorrMaskTask(RegisteredTask):
  def __init__(self, cv_list, dst_pre, dst_post, z_list, dst_z, bbox, mip, 
               operators, threshold, dilate_radius=0):
    super().__init__(cv_list, dst_pre, dst_post, z_list, dst_z, bbox, mip, 
                     operators, threshold, dilate_radius)

  def execute(self, aligner):
    cv_list = [DCV(f) for f in self.cv_list]
    dst_pre = DCV(self.dst_pre)
    dst_post = DCV(self.dst_post)
    z_list = self.z_list
    dst_z = self.dst_z
    patch_bbox = deserialize_bbox(self.bbox)
    mip = self.mip
    operators = self.operators
    threshold = self.threshold
    dilate_radius = self.dilate_radius
    print("\nFcorrMaskTask\n"
          "cv_list {}\n"
          "dst_pre {}\n"
          "dst_post {}\n"
          "z_list {}\n"
          "dst_z {}\n"
          "MIP{}\n"
          "operators {}\n"
          "threshold {}\n"
          "dilate_radius {}\n"
          .format(cv_list, dst_pre, dst_post, z_list, dst_z, mip, operators, 
                  threshold, dilate_radius),
          flush=True)
    start = time()
    images = []
    for cv, z in zip(cv_list, z_list):
      image = aligner.get_data(cv, z, patch_bbox, src_mip=mip, dst_mip=mip,
                            to_float=False, to_tensor=True)
      images.append(image)
    cjn = fcorr_conjunction(images, operators)
    aligner.save_image(cjn.numpy(), dst_pre, dst_z, patch_bbox, mip, to_uint8=False)
    mask = (cjn > threshold).numpy()
    if dilate_radius > 0: 
      s = np.ones((dilate_radius, dilate_radius), dtype=bool)
      mask = ndimage.binary_dilation(mask[0,0,...], structure=s).astype(mask.dtype)
      mask = mask[np.newaxis, np.newaxis, ...]
    aligner.save_image(mask, dst_post, dst_z, patch_bbox, mip, to_uint8=True)
    end = time()
    diff = end - start
    print('FcorrMaskTask: {:.3f} s'.format(diff))

class MaskLogicTask(RegisteredTask):
  def __init__(self, cv_list, dst_cv, z_list, dst_z, bbox, mip_list, dst_mip, op):
    super(). __init__(cv_list, dst_cv, z_list, dst_z, bbox, mip_list, dst_mip, op)

  def execute(self, aligner):
    cv_list = [DCV(f) for f in self.cv_list]
    dst = DCV(self.dst_cv)
    z_list = self.z_list
    dst_z = self.dst_z
    patch_bbox = deserialize_bbox(self.bbox)
    mip_list = self.mip_list
    dst_mip = self.dst_mip
    op = self.op
    print("\nMaskLogicTask\n"
          "op {}\n"
          "cv_list {}\n"
          "dst {}\n"
          "z_list {}\n"
          "dst_z {}\n"
          "mip_list {}\n"
          "dst_mip {}\n"
          .format(op, cv_list, dst, z_list, dst_z, mip_list, dst_mip),
          flush=True)
    start = time()
    if op == 'and':
      res = aligner.mask_conjunction_chunk(cv_list, z_list, patch_bbox, mip_list,
                                           dst_mip)
    elif op == 'or':
      res = aligner.mask_disjunction_chunk(cv_list, z_list, patch_bbox, mip_list,
                                           dst_mip)

    aligner.save_image(res, dst, dst_z, patch_bbox, dst_mip, to_uint8=True)
    end = time()
    diff = end - start
    print('Task: {:.3f} s'.format(diff))

class MaskOutTask(RegisteredTask):
  def __init__(self, cv, mip, z, bbox):
    super(). __init__(cv, mip, z, bbox)

  def execute(self, aligner):
    cv = DCV(self.cv)
    mip = self.mip
    z = self.z
    bbox = deserialize_bbox(self.bbox)
    mask = aligner.get_ones(bbox, mip)
    mask = mask[np.newaxis,np.newaxis,...] 
    aligner.save_image(mask, cv, z, bbox, mip, to_uint8=True)
    print('Mask out: section {}'.format(z))

class ComputeFcorrTask(RegisteredTask):
  def __init__(self, src_cv, dst_pre_cv, dst_post_cv, patch_bbox, src_mip, dst_mip,
               src_z, tgt_z, dst_z, chunk_size, fill_value, preprocessor_path,
               ignore_pix_val=None,
               lower_bound=None, upper_bound=None,
               mask_cv=None, mask_mip=None, mask_val=None):
    super(). __init__(src_cv, dst_pre_cv, dst_post_cv, patch_bbox, src_mip, dst_mip,
                      src_z, tgt_z, dst_z, chunk_size, fill_value,
                      preprocessor_path, ignore_pix_val,
                      lower_bound, upper_bound,
                      mask_cv, mask_mip, mask_val)

  def execute(self, aligner):
    src_cv = DCV(self.src_cv)
    dst_pre_cv = DCV(self.dst_pre_cv)
    dst_post_cv = DCV(self.dst_post_cv)
    src_z = self.src_z
    tgt_z = self.tgt_z
    dst_z = self.dst_z
    patch_bbox = deserialize_bbox(self.patch_bbox)
    src_mip = self.src_mip
    dst_mip = self.dst_mip
    chunk_size = self.chunk_size
    fill_value = self.fill_value
    preprocessor_path = self.preprocessor_path
    ignore_pix_val = self.ignore_pix_val
    lower_bound = self.lower_bound or (0.0, 0.0)
    upper_bound = self.upper_bound or (1.0, 1.0)
    if self.mask_cv:
      mask_cv = DCV(self.mask_cv)
    else:
      mask_cv = None
    mask_mip = self.mask_mip
    mask_val = self.mask_val
    print("\nFCorr"
          "src_cv {}\n"
          "dst_pre_cv {}\n"
          "dst_post_cv {}\n"
          "src_z={} to tgt_z={}\n"
          "dst_z={}\n"
          "src_mip={}, dst_mip={}\n"
          "chunk_size={}\n"
          "fill_value={}\n"
          "preprocessor={}\n"
          "ignoring pixvals: {}\n"
          "lower_bound={} -> {}\n"
          "upper_bound={} -> {}\n"
          "mask_cv {}, mip {}, val {}"
          "\n".format(src_cv, dst_pre_cv, dst_post_cv, src_z, tgt_z, dst_z, src_mip, 
                      dst_mip, chunk_size, fill_value, preprocessor_path, ignore_pix_val, 
                      lower_bound[0], lower_bound[1],
                      upper_bound[0], upper_bound[1],
                      mask_cv, mask_mip, mask_val), flush=True)
    start = time()
    post_image, pre_image = aligner.get_fcorr(src_cv, src_z, tgt_z, patch_bbox, src_mip,
                                              chunk_size, fill_value,
                                              preprocessor_path,
                                              ignore_pix_val,
                                              lower_bound=lower_bound,
                                              upper_bound=upper_bound,
                                              mask_cv=mask_cv,
                                              mask_mip=mask_mip,
                                              mask_val=mask_val)
    aligner.save_image(pre_image, dst_pre_cv, dst_z, patch_bbox, dst_mip, to_uint8=False)
    aligner.save_image(post_image, dst_post_cv, dst_z, patch_bbox, dst_mip, 
                       to_uint8=False)
    end = time()
    diff = end - start
    print('FcorrTask: {:.3f} s'.format(diff))

class Dilation(RegisteredTask):
  """Binary dilation only, right now
  """
  def __init__(self, src_cv, dst_cv, src_z, dst_z, bbox, mip, 
               radius):
    super(). __init__(src_cv, dst_cv, src_z, dst_z, bbox, mip, 
                      radius)

  def execute(self, aligner):
    src_cv = DCV(self.src_cv)
    dst_cv = DCV(self.dst_cv)
    src_z = self.src_z
    dst_z = self.dst_z
    bbox = deserialize_bbox(self.bbox)
    mip = self.mip
    radius = self.radius
    print("\nDilation"
          "src_cv {}\n"
          "dst_cv {}\n"
          "src_z {}, dst_z {}\n"
          "mip {}\n"
          "radius {}\n"
          .format(src_cv, dst_cv, src_z, dst_z, mip, radius), 
          flush=True)
    start = time()
    pad = (radius - 1) // 2 
    padded_bbox = deepcopy(bbox)
    padded_bbox.max_mip = mip
    padded_bbox.uncrop(pad, mip=mip)
    d = aligner.get_data(src_cv, src_z, padded_bbox, src_mip=mip, dst_mip=mip,
                         to_float=True, to_tensor=True)
    assert(radius > 0)
    s = torch.ones((1,1,radius,radius), device=d.device)
    o = conv2d(d, s) > 0
    if o.is_cuda:
      o = o.data.cpu()
    o = o.numpy()
    aligner.save_image(o, dst_cv, dst_z, bbox, mip, to_uint8=True)
    end = time()
    diff = end - start
    print('Dilation: {:.3f} s'.format(diff))

class Threshold(RegisteredTask):
  def __init__(self, src_cv, dst_cv, src_z, dst_z, bbox, mip, 
               threshold, op):
    super(). __init__(src_cv, dst_cv, src_z, dst_z, bbox, mip, 
                      threshold, op)

  def execute(self, aligner):
    src_cv = DCV(self.src_cv)
    dst_cv = DCV(self.dst_cv)
    src_z = self.src_z
    dst_z = self.dst_z
    bbox = deserialize_bbox(self.bbox)
    mip = self.mip
    threshold = self.threshold
    op = self.op
    print("\nThreshold"
          "src_cv {}\n"
          "dst_cv {}\n"
          "src_z {}, dst_z {}\n"
          "mip {}\n"
          "img {} {}\n"
          .format(src_cv, dst_cv, src_z, dst_z, mip, op, threshold), 
          flush=True)
    fn_lookup = {'>': operator.gt, 
                 '>=': operator.ge, 
                 '<': operator.lt, 
                 '<=': operator.le,
                 '==': operator.eq,
                 '!=': operator.ne} 
    start = time()
    assert(op in fn_lookup)
    fn = fn_lookup[op] 
    d = aligner.get_data(src_cv, src_z, bbox, src_mip=mip, dst_mip=mip,
                         to_float=False, to_tensor=True)
    o = fn(d, threshold)
    if o.is_cuda:
      o = o.data.cpu()
    o = o.numpy()
    aligner.save_image(o, dst_cv, dst_z, bbox, mip, to_uint8=True)
    end = time()
    diff = end - start
    print('Threshold: {:.3f} s'.format(diff))

class ComputeSmoothness(RegisteredTask):
  def __init__(self, src_cv, dst_cv, src_z, dst_z, bbox, 
               mip):
    super(). __init__(src_cv, dst_cv, src_z, dst_z, bbox, 
                      mip)

  def execute(self, aligner):
    src_cv = DCV(self.src_cv)
    dst_cv = DCV(self.dst_cv)
    src_z = self.src_z
    dst_z = self.dst_z
    bbox = deserialize_bbox(self.bbox)
    mip = self.mip
    print("\nComputeSmoothness"
          "src_cv {}\n"
          "dst_cv {}\n"
          "src_z {}, dst_z {}\n"
          "mip {}\n"
          .format(src_cv, dst_cv, src_z, dst_z, mip), flush=True)
    start = time()
    pad = 256 
    penalty = aligner.compute_smoothness_chunk(src_cv, src_z, bbox, mip, pad)
    penalty = penalty.data.cpu().numpy()
    aligner.save_image(penalty[:,:,pad:-pad,pad:-pad], dst_cv, dst_z, bbox, mip, 
                       to_uint8=False)
    end = time()
    diff = end - start
    print('ComputeSmoothness: {:.3f} s'.format(diff))

class SumPoolTask(RegisteredTask):
  def __init__(self, src_cv, dst_cv, src_z, dst_z, bbox, 
               src_mip, dst_mip):
    super(). __init__(src_cv, dst_cv, src_z, dst_z, bbox, 
                      src_mip, dst_mip)

  def execute(self, aligner):
    src_cv = DCV(self.src_cv)
    dst_cv = DCV(self.dst_cv)
    src_z = self.src_z
    dst_z = self.dst_z
    bbox = deserialize_bbox(self.bbox)
    src_mip = self.src_mip
    dst_mip = self.dst_mip
    print("\nSumPool"
          "src_cv {}\n"
          "dst_cv {}\n"
          "src_z {}, dst_z {}\n"
          "src_mip {}\n"
          "dst_mip {}\n"
          .format(src_cv, dst_cv, src_z, dst_z, src_mip, dst_mip), flush=True)
    start = time()
    chunk_dim = (2**(dst_mip - src_mip), 2**(dst_mip - src_mip))
    sum_pool = LPPool2d(1, chunk_dim, stride=chunk_dim).to(device=aligner.device)
    d = aligner.get_data(src_cv, src_z, bbox, src_mip=src_mip, dst_mip=src_mip,
                        to_float=False, to_tensor=True).float()
    o = sum_pool(d)
    if o.is_cuda:
      o = o.data.cpu()
    o = o.numpy()
    aligner.save_image(o, dst_cv, dst_z, bbox, dst_mip, to_uint8=False)
    end = time()
    diff = end - start
    print('SumPool: {:.3f} s'.format(diff))

class SummarizeTask(RegisteredTask):
  def __init__(self, src_cv, dst_path, z, bbox, mip):
    super(). __init__(src_cv, dst_path, z, bbox, mip)

  def execute(self, aligner):
    src_cv = DCV(self.src_cv)
    dst_path = self.dst_path 
    z = self.z
    bbox = deserialize_bbox(self.bbox)
    mip = self.mip
    print("\nSummarize"
          "src_cv {}\n"
          "dst_path {}\n"
          "z {}\n"
          "mip {}\n"
          .format(src_cv, dst_path, z, mip), flush=True)
    start = time()
    d = aligner.get_data(src_cv, z, bbox, src_mip=mip, dst_mip=mip,
                         to_float=False, to_tensor=False)
    stats = {}
    stats['sum'] = float(np.sum(d))
    stats['std'] = float(np.std(d))
    stats['count'] = float(np.prod(d.shape))
    stats['min'] = float(np.min(d))
    stats['max'] = float(np.max(d))
    stats['mean'] = float(np.mean(d))
    stats['med'] = float(np.median(d))
    print(stats)
    with Storage(dst_path) as stor:
      path = '{}/{}'.format(bbox.stringify(0), z)
      stor.put_file(path, json.dumps(stats),
                    content_type='application/json',
                    cache_control='no-cache')
      print('Save summary at {}'.format(join(dst_path, path)))
    end = time()
    diff = end - start
    print('SummarizeTask: {:.3f} s'.format(diff))
