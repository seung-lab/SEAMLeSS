import boto3
import time
import json
import tenacity

from .boundingbox import BoundingBox, deserialize_bbox

from taskqueue import RegisteredTask

class BatchRenderTask(RegisteredTask):
  def __init__(
    self, z, field_cv, field_z, patches, 
    mip, dst_cv, dst_z, batch
  ):
    super().__init__(
      z, field_cv, field_z, patches, 
      mip, dst_cv, dst_z, batch
    )
    self.patches = [p.serialize() for p in patches]

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

class ComposeTask(RegisteredTask):
  def __init__(
    self, z, coarse_cv, fine_cv, dst_cv, 
    bbox, coarse_mip, fine_mip
  ):
    super().__init__(
      z, coarse_cv, fine_cv, dst_cv, 
      bbox, coarse_mip, fine_mip
    )

  def execute(self, aligner):
    z = self.z
    coarse_cv = DCV(self.coarse_cv)
    fine_cv = DCV(self.fine_cv)
    dst_cv = DCV(self.dst_cv)
    bbox = deserialize_bbox(self.bbox)
    coarse_mip = self.coarse_mip
    fine_mip = self.fine_mip
    h = aligner.compose_cloudvolumes(z, fine_cv, coarse_cv, bbox, fine_mip, coarse_mip)        
    aligner.save_vector_patch(dst_cv, z, h, bbox, fine_mip)    

class CopyTask(RegisteredTask):
  def __init__(self, z, dst_cv, dst_z, patches, mip):
    super().__init__(z, dst_cv, dst_z, patches, mip)
    self.patches = [p.serialize() for p in patches]

  def execute(self, aligner):
    z = self.z
    patches  = [deserialize_bbox(p) for p in self.patches]
    mip = self.mip
    dst_cv = DCV(self.dst_cv)
    dst_z = self.dst_z
    
    def chunkwise(patch_bbox):
      src_cv = aligner.src['src_img']
      if 'src_mask' in aligner.src:
        mask_cv = aligner.src['src_mask']
        raw_patch = aligner.get_image(src_cv, z, patch_bbox, mip,
                                    adjust_contrast=False, to_tensor=True)
        raw_mask = aligner.get_mask(mask_cv, z, patch_bbox, 
                                 src_mip=aligner.src.src_mask_mip,
                                 dst_mip=mip, valid_val=aligner.src.src_mask_val)
        raw_patch = raw_patch.masked_fill_(raw_mask, 0)
        raw_patch = raw_patch.cpu().numpy()
      else: 
        raw_patch = aligner.get_image(src_cv, z, patch_bbox, mip,
                                    adjust_contrast=False, to_tensor=False)
      aligner.save_image_patch(dst_cv, dst_z, raw_patch, patch_bbox, mip)
    
    aligner.pool.map(chunkwise, patches)    

class DownsampleTask(RegisteredTask):
  def __init__(self, cv, z, patches, mip):
    super().__init__(cv, z, patches, mip)
    self.patches = [p.serialize() for p in patches]

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
    self.patches = [ p.serialize() for p in patches ]

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

class RenderTask(RegisteredTask):
  def __init__(self, z, field_cv, field_z, patches, mip, dst_cv, dst_z):
    super().__init__(z, field_cv, field_z, patches, mip, dst_cv, dst_z)
    self.patches = [p.serialize() for p in patches]

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
      warped_patch = aligner.warp_patch(src_z, field_cv, field_z, patch_bbox, mip)
      aligner.save_image_patch(dst_cv, dst_z, warped_patch, patch_bbox, mip)
    aligner.pool.map(chunkwise, patches)
    #warped_patch = aligner.warp_patch(src_z, field_cv, field_z, patches, mip)
    #aligner.save_image_patch(dst_cv, dst_z, warped_patch, patches, mip)    

class RenderCVTask(RegisteredTask):
  def __init__(self, z, field_cv, field_z, patches, mip, dst_cv, dst_z):
    super().__init__(z, field_cv, field_z, patches, mip, dst_cv, dst_z)
    self.patches = [p.serialize() for p in patches]

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
    self.patches = [p.serialize() for p in patches]

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

def ResidualTask(RegisteredTask):
  def __init__(self, src_z, src_cv, tgt_z, tgt_cv, field_cv, patch_bbox, mip):
    super().__init__(src_z, src_cv, tgt_z, tgt_cv, field_cv, patch_bbox, mip)

  def execute(self, aligner):
    src_cv = DCV(self.src_cv) 
    tgt_cv = DCV(self.tgt_cv) 
    field_cv = DCV(self.field_cv) 
    patch_bbox = deserialize_bbox(self.patch_bbox)
    mip = self.mip

    aligner.compute_residual_patch(
      self.src_z, src_cv, self.tgt_z, tgt_cv, 
      field_cv, patch_bbox, mip
    )

def ResAndComposeTask(RegisteredTask):
  def __init__(self, z, forward, reverse, patch_bbox, mip, w_cv):
    super().__init__(z, forward, reverse, patch_bbox, mip, w_cv)

  def execute(self, aligner):
    patch_bbox = deserialize_bbox(self.patch_bbox)
    w_cv = DCV(self.w_cv)
    aligner.res_and_compose(
      self.z, self.forward, self.reverse, 
      patch_bbox, self.mip, w_cv
    )

class UpsampleRenderRechunkTask(RegisteredTask):
  def __init__(
    self, z_range, src_cv, field_cv, dst_cv, 
    patches, image_mip, field_mip
  ):
    super().__init__(
      z_range, src_cv, field_cv, dst_cv, 
      patches, image_mip, field_mip
    )
    self.patches = [p.serialize() for p in patches]

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

class VectorVoteTask(RegisteredTask):
  def __init__(
    self, z_range, read_F_cv, write_F_cv, 
    patch_bbox, mip, inverse, T, negative_offsets, 
    serial_operation
  ):
    super().__init__(
      z_range, read_F_cv, write_F_cv, patch_bbox, mip, 
      inverse, T, negative_offsets, serial_operation
    )

  def execute(self, aligner):
    read_F_cv = DCV(self.read_F_cv)
    write_F_cv =DCV(self.write_F_cv)
    chunks = deserialize_bbox(self.patch_bbox)
    z_range = range(self.z_start, self.z_end+1)

    aligner.vector_vote(
      z_range, read_F_cv, write_F_cv, chunks, 
      self.mip, inverse=self.inverse, T=self.T, 
      negative_offsets=self.negative_offsets, 
      serial_operation=self.serial_operation
    ) 