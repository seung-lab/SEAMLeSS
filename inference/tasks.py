import boto3
import time
import json
import tenacity
from mipless_cloudvolume import deserialize_miplessCV as DCV
from cloudvolume import Storage
from cloudvolume.lib import scatter 
from boundingbox import BoundingBox, deserialize_bbox

from taskqueue import RegisteredTask, TaskQueue, LocalTaskQueue
# from taskqueue.taskqueue import _scatter as scatter

def run(aligner, tasks): 
  if aligner.distributed:
    TQ = TaskQueue(queue_name=aligner.queue_name)
  else:
    TQ = LocalTaskQueue(queue_name=aligner.queue_name, parallel=1)

  def multiprocess_upload(ptasks):
    with TQ as tq:
      for task in ptasks:
        tq.insert(task, args=[ aligner ])
  
  if aligner.threads == 1:
    multiprocess_upload(tasks)
  else:
    tasks = scatter(tasks, aligner.threads)
    with concurrent.futures.ProcessPoolExecutor(max_workers=aligner.threads) as executor:
      executor.map(multiprocess_upload, tasks)

class CopyTask(RegisteredTask):
  def __init__(self, src_cv, dst_cv, src_z, dst_z, patch_bbox, mip, 
               is_field, mask_cv, mask_mip, mask_val, prefix):
    super().__init__(src_cv, dst_cv, src_z, dst_z, patch_bbox, mip, 
                     is_field, mask_cv, mask_mip, mask_val, prefix)

  def execute(self, aligner):
    src_cv = DCV(self.src_cv)
    dst_cv = DCV(self.dst_cv)
    src_z = self.src_z
    dst_z = self.dst_z
    patch_bbox = deserialize_bbox(self.patch_bbox)
    mip = self.mip
    is_field = self.is_field
    mask_cv = None 
    if self.mask_cv:
      mask_cv = DCV(self.mask_cv)
    mask_mip = self.mask_mip
    mask_val = self.mask_val
    prefix = self.prefix
    print("\nCopy\n"
          "src {}\n"
          "dst {}\n"
          "z={} to z={}\n"
          "MIP{}\n".format(src_cv, dst_cv, src_z, dst_z, mip), flush=True)

    if not aligner.dry_run:
      if is_field:
        field =  aligner.get_field(src_cv, src_z, patch_bbox, mip, relative=False,
                                to_tensor=False)
        aligner.save_field(field, dst_cv, dst_z, patch_bbox, mip, relative=False)
      else:
        image = aligner.get_masked_image(src_cv, src_z, patch_bbox, mip,
                                mask_cv=mask_cv, mask_mip=mask_mip,
                                mask_val=mask_val,
                                to_tensor=False, normalizer=None)
        aligner.save_image(image, dst_cv, dst_z, patch_bbox, mip)
      with Storage(dst_cv.path) as stor:
          path = 'copy_done/{}/{}'.format(prefix, patch_bbox.stringify(dst_z))
          stor.put_file(path, '')
          print('Marked finished at {}'.format(path))

class ComputeFieldTask(RegisteredTask):
  def __init__(self, model_path, src_cv, tgt_cv, field_cv, src_z, tgt_z, 
                     patch_bbox, mip, pad, prefix):
    super().__init__(model_path, src_cv, tgt_cv, field_cv, src_z, tgt_z, 
                                 patch_bbox, mip, pad, prefix)

  def execute(self, aligner):
    model_path = self.model_path
    src_cv = DCV(self.src_cv) 
    tgt_cv = DCV(self.tgt_cv) 
    field_cv = DCV(self.field_cv) 
    src_z = self.src_z
    tgt_z = self.tgt_z
    patch_bbox = deserialize_bbox(self.patch_bbox)
    mip = self.mip
    pad = self.pad
    prefix = self.prefix
    print("\nCompute field\n"
          "model {}\n"
          "src {}\n"
          "tgt {}\n"
          "field {}\n"
          "z={} to z={}\n"
          "MIP{}\n".format(model_path, src_cv, tgt_cv, field_cv, src_z, tgt_z,
                           mip), flush=True)
    if not aligner.dry_run:
      field = aligner.compute_field_chunk(model_path, src_cv, tgt_cv, src_z, tgt_z, 
      			 patch_bbox, mip, pad)
      field = field.data.cpu().numpy()
      aligner.save_field(field, field_cv, src_z, patch_bbox, mip, relative=False)
      with Storage(field_cv.path) as stor:
        path = 'compute_field_done/{}/{}'.format(prefix, patch_bbox.stringify(src_z))
        stor.put_file(path, '')
        print('Marked finished at {}'.format(path))

class RenderTask(RegisteredTask):
  def __init__(self, src_cv, field_cv, dst_cv, src_z, field_z, dst_z, patch_bbox, src_mip, 
                     field_mip, mask_cv, mask_mip, mask_val, prefix):
    super(). __init__(src_cv, field_cv, dst_cv, src_z, field_z, dst_z, patch_bbox, src_mip, 
                     field_mip, mask_cv, mask_mip, mask_val, prefix)

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
    prefix = self.prefix
    print("\nRendering\n"
          "src {0}\n"
          "field {1}\n"
          "dst {2}\n"
          "z={3} to z={4}\n"
          "MIP{5} to MIP{6}\n".format(src_cv, field_cv, dst_cv, 
                              src_z, dst_z, field_mip, src_mip), flush=True)
    if not aligner.dry_run:
      image = aligner.cloudsample_image(src_cv, field_cv, src_z, field_z, 
                                     patch_bbox, src_mip, field_mip, 
                                     mask_cv=mask_cv, mask_mip=mask_mip,
                                     mask_val=mask_val)
      image = image.cpu().numpy()
      aligner.save_image(image, dst_cv, dst_z, patch_bbox, src_mip)
      with Storage(dst_cv.path) as stor:
        path = 'render_done/{}/{}'.format(prefix, patch_bbox.stringify(dst_z))
        stor.put_file(path, '')
        print('Marked finished at {}'.format(path))

class VectorVoteTask(RegisteredTask):
  def __init__(self, pairwise_cvs, vvote_cv, z, patch_bbox, mip, inverse, 
                     softmin_temp, serial, prefix):
    super().__init__(pairwise_cvs, vvote_cv, z, patch_bbox, mip, inverse,
                     softmin_temp, serial, prefix)

  def execute(self, aligner):
    pairwise_cvs = {int(k): DCV(v) for k,v in self.pairwise_cvs.items()}
    vvote_cv = DCV(self.vvote_cv)
    z = self.z
    patch_bbox = deserialize_bbox(self.patch_bbox)
    mip = self.mip
    inverse = bool(self.inverse)
    softmin_temp = self.softmin_temp
    serial = bool(self.serial)
    prefix = self.prefix
    print("\nVector vote\n"
          "fields {}\n"
          "dst {}\n"
          "z={}\n"
          "MIP{}\n"
          "inverse={}\n"
          "softmin_temp={}\n"
          "serial={}\n".format(pairwise_cvs.keys(), vvote_cv, z, 
                              mip, inverse, softmin_temp, serial),
                              flush=True)
    if not aligner.dry_run:
      field = aligner.vector_vote_chunk(pairwise_cvs, vvote_cv, z, patch_bbox, mip, 
                       inverse=inverse, softmin_temp=softmin_temp, 
                       serial=serial)
      field = field.data.cpu().numpy()
      aligner.save_field(field, vvote_cv, z, patch_bbox, mip, relative=False)
      with Storage(vvote_cv.path) as stor:
        path = 'vector_vote_done/{}/{}'.format(prefix, patch_bbox.stringify(z))
        stor.put_file(path, '')
        print('Marked finished at {}'.format(path))

class ComposeTask(RegisteredTask):
  def __init__(self, f_cv, g_cv, dst_cv, f_z, g_z, dst_z, patch_bbox, f_mip, g_mip, 
                     dst_mip, factor, prefix):
    super().__init__(f_cv, g_cv, dst_cv, f_z, g_z, dst_z, patch_bbox, f_mip, g_mip, 
                     dst_mip, factor, prefix)

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
    prefix = self.prefix
    print("\nCompose\n"
          "f {}\n"
          "g {}\n"
          "f_z={}, g_z={}\n"
          "f_MIP{}, g_MIP{}\n"
          "dst {}\n"
          "dst_MIP {}\n"
          "factor={}\n".format(f_cv, g_cv, f_z, g_z, f_mip, g_mip, dst_cv, 
                               dst_mip, factor), flush=True)
    if not aligner.dry_run:
      h = aligner.get_composed_field(f_cv, g_cv, f_z, g_z, patch_bbox, 
                                   f_mip, g_mip, dst_mip, factor)
      h = h.data.cpu().numpy()
      aligner.save_field(h, dst_cv, dst_z, patch_bbox, dst_mip, relative=False)
      with Storage(dst_cv.path) as stor:
        path = 'compose_done/{}/{}'.format(prefix, patch_bbox.stringify(dst_z))
        stor.put_file(path, '')
        print('Marked finished at {}'.format(path))

class CPCTask(RegisteredTask):
  def __init__(self, src_cv, tgt_cv, dst_cv, src_z, tgt_z, patch_bbox, 
                    src_mip, dst_mip, norm, prefix):
    super().__init__(src_cv, tgt_cv, dst_cv, src_z, tgt_z, patch_bbox, 
                    src_mip, dst_mip, norm, prefix)

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
    prefix = self.prefix
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
      with Storage(dst_cv.path) as stor:
        path = 'cpc_done/{}/{}'.format(prefix, patch_bbox.stringify(src_z))
        stor.put_file(path, '')
        print('Marked finished at {}'.format(path))

class DownsampleTask(RegisteredTask):
  def __init__(self, cv, z, patch_bbox, src_mip, dst_mip, prefix):
    super().__init__(cv, z, patch_bbox, src_mip, dst_mip, prefix)

  def execute(self, aligner):
    cv = DCV(self.cv) 
    z = self.z
    patch_bbox = deserialize_bbox(self.patch_bbox)
    src_mip = self.src_mip
    dst_mip = self.dst_mip
    prefix = self.prefix
    print("\nDownsampling\n"
          "cv {}\n"
          "z {}\n"
          "MIP{} to MIP{}\n".format(cv, z, src_mip, dst_mip), flush=True)
    if not aligner.dry_run:
      image = self.downsample_chunk(cv, z, patch_bbox, src_mip, dst_mip)
      image = image.cpu().numpy()
      aligner.save_image(image, cv, z, patch_bbox, dst_mip)
      with Storage(dst_cv.path) as stor:
        path = 'downsample_done/{}/{}'.format(prefix, patch_bbox.stringify(dst_z))
        stor.put_file(path, '')
        print('Marked finished at {}'.format(path))

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
      with Storage(dst_cv.path) as stor:
          stor.put_file('render_batch/'+str(mip)+'_'+str(dst_z)+'_'+str(batch)+'/'+ patch_bbox.__str__(), '')
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
      with Storage(cv.path) as stor:
          stor.put_file('downsample_done/'+str(mip)+'_'+str(z)+'/'+patch_bbox.__str__(), '')
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
      with Storage(dst_cv.path) as stor:
          stor.put_file('render_cv/'+str(mip)+'_'+str(dst_z)+'/'+ patch_bbox.__str__(), '')
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
      with Storage(dst_cv.path) as stor:
          stor.put_file('render_low_mip/'+str(image_mip)+'_'+str(dst_z)+'/'+ patch_bbox.__str__(), '')
    aligner.pool.map(chunkwise, patches)

class ResAndComposeTask(RegisteredTask):
  def __init__(self, z, forward, reverse, patch_bbox, mip, w_cv):
    super().__init__(z, forward, reverse, patch_bbox, mip, w_cv)

  def execute(self, aligner):
    patch_bbox = deserialize_bbox(self.patch_bbox)
    w_cv = DCV(self.w_cv)
    aligner.res_and_compose(
      self.z, self.forward, self.reverse, 
      patch_bbox, self.mip, w_cv
    )
    with Storage(w_cv.path) as stor:
        stor.put_file('res_and_compose/'+str(self.mip)+'/'+str(self.z)+'/'+patch_bbox.__str__(), '')

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

