import boto3
from time import time
import json
import tenacity
import numpy as np
from functools import partial
from mipless_cloudvolume import deserialize_miplessCV as DCV
from cloudvolume import Storage
from cloudvolume.lib import scatter 
from boundingbox import BoundingBox, deserialize_bbox
import csv
import os

from taskqueue import RegisteredTask, TaskQueue, LocalTaskQueue, GreenTaskQueue
from concurrent.futures import ProcessPoolExecutor
# from taskqueue.taskqueue import _scatter as scatter

tmp_dir= "/tmp/alignment/"

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

def init_checkpoint():
    os.system("rm -rf " + tmp_dir)
    os.system("mkdir -p " + tmp_dir)
    os.system("mkdir -p " + tmp_dir +"img/")

def print_obj(member_dict, classname):
    member_dict["class"] = classname
    with open(tmp_dir+"name","w") as f:
        json.dump(member_dict,f)

    #w = csv.writer(open(tmp_dir+"name", "w"))
    #w.writerow(["class", classname])
    #for key, val in member_dict.items():
    #    w.writerow([key, val])


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
  def __init__(self, model_path, src_cv, dst_cv, z, mip, bbox, prefix):
    super().__init__(model_path, src_cv, dst_cv, z, mip, bbox, prefix)

  def execute(self, aligner):
    src_cv = DCV(self.src_cv)
    dst_cv = DCV(self.dst_cv)
    z = self.z
    patch_bbox = deserialize_bbox(self.bbox)
    mip = self.mip
    prefix = self.prefix
    print("\nPredict Image\n"
          "src {}\n"
          "dst {}\n"
          "at z={}\n"
          "MIP{}\n".format(src_cv, dst_cv, z, mip), flush=True)
    start = time()
    image = aligner.predict_image_chunk(self.model_path, src_cv, z, mip, patch_bbox)
    image = image.cpu().numpy()
    aligner.save_image(image, dst_cv, z, patch_bbox, mip)

    with Storage(dst_cv.path) as stor:
        path = 'predict_image_done/{}/{}'.format(prefix, patch_bbox.stringify(z))
        stor.put_file(path, '')
        print('Marked finished at {}'.format(path))
    end = time()
    diff = end - start
    print(':{:.3f} s'.format(diff))

class StitchComposeRenderTask(RegisteredTask):
    def __init__(self, qu, influence_index, z_start, z_stop, b_field,
                 influence_blocks, src,
                 vv_field_cv, decay_dist, src_mip, dst_mip, bbox, pad,
                 extra_off, finish_dir, chunk_size, dst, upsample_mip,
                 upsample_bbox):
        super().__init__(qu, influence_index, z_start, z_stop, b_field,
                         influence_blocks, src,
                         vv_field_cv, decay_dist, src_mip, dst_mip, bbox, pad,
                         extra_off, finish_dir, chunk_size, dst, upsample_mip,
                         upsample_bbox)
    def execute(self, aligner):
        init_checkpoint()
        print_obj(self._args, "StitchComposeRender")
        tq = TaskQueue(self.qu)
        tq.delete(self._id)
        z_range = range(self.z_start, self.z_stop)
        b_field = DCV(self.b_field)
        influence_blocks = self.influence_blocks
        src = DCV(self.src)
        vv_field_cv = DCV(self.vv_field_cv)
        decay_dist = self.decay_dist
        src_mip = self.src_mip
        bbox = deserialize_bbox(self.bbox)
        dst_mip = self.dst_mip
        pad = self.pad
        extra_off = self.extra_off
        chunk_size = self.chunk_size
        dst = DCV(self.dst)
        upsample_mip = self.upsample_mip
        upsample_bbox = deserialize_bbox(self.upsample_bbox)
        print("\n Stitch compose and render task\n"
              "src {}\n"
              "MIP{}\n"
              "start_z={} \n".format(self.src, src_mip,
                                     z_range.start), flush=True)
        start = time()
        aligner.stitch_compose_render(z_range, b_field, influence_blocks, src, vv_field_cv,
                                      decay_dist, src_mip, dst_mip, bbox, pad, extra_off,
                                      chunk_size, dst, self.finish_dir,
                                      upsample_mip, upsample_bbox)
        with Storage(self.dst) as stor:
            path = 'stitch_rander_done/{}/{}'.format(str(dst_mip),
                                                        str(self.influence_index))
            stor.put_file(path, '')
            print('Marked finished at {}'.format(path))
        end = time()
        diff = end - start
        print('Stitch compose and render task time:{:.3f} s'.format(diff), flush=True)

class StitchGetField(RegisteredTask):
    def __init__(self, qu, param_lookup, bs, be, src_cv, tgt_cv, prev_field_cv,
                 bfield_cv, tmp_img_cv, tmp_vvote_field_cv, mip, pad, bbox,
                 start_z, finish_dir, chunk_size, softmin_temp, blur_sigma):
        super().__init__(qu, param_lookup, bs, be, src_cv, tgt_cv, prev_field_cv,
                         bfield_cv, tmp_img_cv, tmp_vvote_field_cv, mip, pad,
                         bbox, start_z, finish_dir, chunk_size,
                         softmin_temp, blur_sigma)
    def execute(self, aligner):
        init_checkpoint()
        print_obj(self._args, "StitchGetField")
        tq = TaskQueue(self.qu)
        tq.delete(self._id)
        src_cv = DCV(self.src_cv)
        tgt_cv = DCV(self.tgt_cv)
        prev_field_cv = DCV(self.prev_field_cv)
        bfield_cv = DCV(self.bfield_cv)
        param_lookup= self.param_lookup
        mip = self.mip
        bbox = deserialize_bbox(self.bbox)
        chunk_size = self.chunk_size
        softmin_temp = self.softmin_temp
        blur_sigma = self.blur_sigma
        pad = self.pad
        bs = self.bs
        be = self.be
        start_z = self.start_z
        tmp_img_cv = DCV(self.tmp_img_cv)
        tmp_vvote_field_cv = DCV(self.tmp_vvote_field_cv)
        print("\n Stitch get field task\n"
              "src {}\n"
              "MIP{}\n"
              "start_z={} \n".format(self.src_cv, mip,
                                     bs), flush=True)
        start = time()

        aligner.get_stitch_field_task(param_lookup, bs, be, src_cv, tgt_cv, prev_field_cv,
                            bfield_cv, tmp_img_cv, tmp_vvote_field_cv, mip, bbox, chunk_size,
                                      pad, start_z, self.finish_dir, softmin_temp, blur_sigma)
        with Storage(self.bfield_cv) as stor:
            path = 'get_stitch_field_done/{}/{}'.format(str(mip), str(bs))
            stor.put_file(path, '')
            print('Marked finished at {}'.format(path))

        end = time()
        diff = end - start
        print('Stitch get field task time:{:.3f} s'.format(diff), flush=True)


class NewAlignTask(RegisteredTask):
  def __init__(self, src, qu, dst, s_field, vvote_field, chunk_grid, mip, pad, block_start,
               block_stop, start_z, chunk_size, model_lookup, finish_dir, mask_cv, mask_mip,
               mask_val):
    super().__init__(src, qu, dst, s_field, vvote_field, chunk_grid, mip, pad, block_start,
                     block_stop, start_z, chunk_size, model_lookup, finish_dir, mask_cv, mask_mip,
                     mask_val)

  def execute(self, aligner):
    init_checkpoint()
    print_obj(self._args, "NewAlignTask")
    tq = TaskQueue(self.qu)
    tq.delete(self._id)
    src_cv = DCV(self.src)
    dst_cv = DCV(self.dst)
    v_field = DCV(self.vvote_field)
    s_field = DCV(self.s_field)
    chunk_grid =[]
    for i in self.chunk_grid:
        print(i)
        chunk_grid.append(deserialize_bbox(i))
    mip = self.mip
    pad = self.pad
    block_start = self.block_start
    block_stop = self.block_stop
    start_z = self.start_z
    chunk_size = self.chunk_size
    model_lookup = self.model_lookup
    mask_cv = None
    if self.mask_cv:
      mask_cv = DCV(self.mask_cv)
    mask_mip = self.mask_mip
    mask_val = self.mask_val
    print("\n Align task\n"
          "src {}\n"
          "MIP{}\n"
          "block_start={} \n".format(self.src, mip,
                        block_start), flush=True)
    start = time()
    aligner.new_align(src_cv, dst_cv, s_field, v_field, chunk_grid, mip, pad, block_start,
                      block_stop, start_z, chunk_size, model_lookup,
                      self.finish_dir,
                      src_mask_cv=mask_cv,
                      src_mask_mip=mask_mip, src_mask_val=mask_val)
    with Storage(self.vvote_field) as stor:
        path = 'block_alignment_done/{}/{}'.format(str(mip), str(block_start))
        stor.put_file(path, '')
        print('Marked finished at {}'.format(path))
    end = time()
    diff = end - start
    print('Align task time:{:.3f} s'.format(diff), flush=True)

class LoadImageTask(RegisteredTask):
  def __init__(self, src_cv, dst_cv, src_z, patch_bbox, mip, step, mask_cv, mask_mip,
               mask_val):
    super().__init__(src_cv, dst_cv, src_z, patch_bbox, mip, step, mask_cv, mask_mip,
               mask_val)

  def execute(self, aligner):
    src_cv = DCV(self.src_cv)
    dst_cv = DCV(self.dst_cv)
    src_z = self.src_z
    patch_bbox = deserialize_bbox(self.patch_bbox)
    mip = self.mip
    mask_cv = None
    if self.mask_cv:
      mask_cv = DCV(self.mask_cv)
    mask_mip = self.mask_mip
    mask_val = self.mask_val
    prefix = mip
    image = []
    for i in range(self.step):
        load_z = src_z +i
        print("\nLoad image from\n"
              "src {}\n"
              "MIP{}\n"
              "z={} \n".format(src_cv, mip,
                                load_z,), flush=True)
        start = time()
        im = aligner.load_part_image(src_cv, load_z, patch_bbox, mip, to_tensor=False)
        image.append(im)
        end = time()
        diff = end - start
        print('load_image time:{:.3f} s'.format(diff), flush=True)
    #with Storage(dst_cv.path) as stor:
    #    path = 'load_image_done/{}/{}'.format(prefix,
    #                                          patch_bbox.stringify(src_z))
    #    stor.put_file(path, '')
    #    print('Marked finished at {}'.format(path))

    #return image

class LoadStoreImageTask(RegisteredTask):
  def __init__(self, src_cv, dst_cv, src_z, patch_bbox, mip, step, mask_cv, mask_mip,
               mask_val, pad, final_chunk, compress):
    super().__init__(src_cv, dst_cv, src_z, patch_bbox, mip, step, mask_cv, mask_mip,
               mask_val, pad, final_chunk, compress)

  def execute(self, aligner):
    src_cv = DCV(self.src_cv)
    dst_cv = DCV(self.dst_cv, compress=self.compress)
    src_z = self.src_z
    patch_bbox = deserialize_bbox(self.patch_bbox)
    final_chunk = deserialize_bbox(self.final_chunk)
    pad = self.pad
    mip = self.mip
    mask_cv = None
    if self.mask_cv:
      mask_cv = DCV(self.mask_cv)
    mask_mip = self.mask_mip
    mask_val = self.mask_val
    prefix = mip
    image = []
    for i in range(self.step):
        load_z = src_z +i
        print("\nLoad image from\n"
              "src {}\n"
              "MIP{}\n"
              "z={} \n".format(src_cv, mip,
                                load_z,), flush=True)
        start = time()
        im = aligner.load_part_image(src_cv, load_z, patch_bbox,
                                     mip, to_tensor=False)
        image.append(im)
        end = time()
        diff = end - start
        print(':{:.3f} s'.format(diff))
    im = image[0][...,pad:-pad, pad:-pad]
    start_save = time()
    aligner.save_image(im, dst_cv, src_z, final_chunk, mip, to_uint8=False)
    write_end = time()
    print("read_time: {} write_time: {}".format(diff,
                                              write_end-start_save),flush=True)

    #with Storage(dst_cv.path) as stor:
    #    path = 'load_image_done/{}/{}'.format(prefix,
    #                                          patch_bbox.stringify(src_z))
    #    stor.put_file(path, '')
    #    print('Marked finished at {}'.format(path))

class RandomStoreImageTask(RegisteredTask):
  def __init__(self, dst_cv, src_z, mip, step, mask_cv, mask_mip,
               mask_val, pad, final_chunk, compress):
    super().__init__(dst_cv, src_z, mip, step, mask_cv, mask_mip,
               mask_val, pad, final_chunk, compress)

  def execute(self, aligner):
    dst_cv = DCV(self.dst_cv, compress=self.compress)
    src_z = self.src_z
    final_chunk = deserialize_bbox(self.final_chunk)
    pad = self.pad
    mip = self.mip
    mask_cv = None
    if self.mask_cv:
      mask_cv = DCV(self.mask_cv)
    mask_mip = self.mask_mip
    mask_val = self.mask_val
    prefix = mip
    x_range = final_chunk.x_range(mip=mip)
    y_range = final_chunk.y_range(mip=mip)
    #im =np.random.randint(255, size=(1,1,x_range[1]-x_range[0],
    #                                 y_range[1]-y_range[0]),
    #                   dtype=np.uint8)
    im = np.ones((1,1,x_range[1]-x_range[0]+2*pad, y_range[1]-y_range[0]+2*pad), dtype=np.uint8)
    #print("image size ", im.nbytes)
    im = im[...,pad:-pad, pad:-pad]
    start_save = time()
    aligner.save_image(im, dst_cv, src_z, final_chunk, mip, to_uint8=False)
    write_end = time()
    print("Random_write_time: {}".format(write_end-start_save),flush=True)



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
      else:
        image = aligner.get_masked_image(src_cv, src_z, patch_bbox, mip,
                                mask_cv=mask_cv, mask_mip=mask_mip,
                                mask_val=mask_val,
                                to_tensor=False, normalizer=None,
                                to_float=False)
        aligner.save_image(image, dst_cv, dst_z, patch_bbox, mip,
                           to_uint8=False)
      with Storage(dst_cv.path) as stor:
          path = 'copy_done/{}/{}'.format(prefix, patch_bbox.stringify(dst_z))
          stor.put_file(path, '')
          print('Marked finished at {}'.format(path))
      end = time()
      diff = end - start
      print(':{:.3f} s'.format(diff))

class ComputeFieldTask(RegisteredTask):
  def __init__(self, model_path, src_cv, tgt_cv, field_cv, src_z, tgt_z, 
                     patch_bbox, mip, pad, src_mask_cv, src_mask_val, src_mask_mip, 
                     tgt_mask_cv, tgt_mask_val, tgt_mask_mip, prefix,
                     prev_field_cv, prev_field_z, prev_field_inverse):
    super().__init__(model_path, src_cv, tgt_cv, field_cv, src_z, tgt_z, 
                     patch_bbox, mip, pad, src_mask_cv, src_mask_val, src_mask_mip, 
                     tgt_mask_cv, tgt_mask_val, tgt_mask_mip, prefix,
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
    prefix = self.prefix
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
    print("other parameters", prev_field_cv,prev_field_z,prev_field_inverse)
    start = time()
    if not aligner.dry_run:
      field = aligner.compute_field_chunk(model_path, src_cv, tgt_cv, src_z, tgt_z, 
                                          patch_bbox, mip, pad, 
                                          src_mask_cv, src_mask_mip, src_mask_val,
                                          tgt_mask_cv, tgt_mask_mip, tgt_mask_val,
                                          None, prev_field_cv, prev_field_z, 
                                          prev_field_inverse)
      aligner.save_field(field, field_cv, src_z, patch_bbox, mip, relative=False)
      with Storage(field_cv.path) as stor:
        path = 'compute_field_done/{}/{}'.format(prefix, patch_bbox.stringify(src_z))
        stor.put_file(path, '')
        print('Marked finished at {}'.format(path))
      end = time()
      diff = end - start
      print('ComputeFieldTask: {:.3f} s'.format(diff))

class RenderTask(RegisteredTask):
  def __init__(self, src_cv, field_cv, dst_cv, src_z, field_z, dst_z, patch_bbox, src_mip,
               field_mip, mask_cv, mask_mip, mask_val, affine, prefix, use_cpu=False):
    super(). __init__(src_cv, field_cv, dst_cv, src_z, field_z, dst_z, patch_bbox, src_mip, 
                     field_mip, mask_cv, mask_mip, mask_val, affine, prefix, use_cpu)

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
    prefix = self.prefix
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
      with Storage(dst_cv.path) as stor:
        path = 'render_done/{}/{}'.format(prefix, patch_bbox.stringify(dst_z))
        stor.put_file(path, '')
        print('Marked finished at {}'.format(path))
      end = time()
      diff = end - start
      print('RenderTask: {:.3f} s'.format(diff))

class VectorVoteTask(RegisteredTask):
  def __init__(self, pairwise_cvs, vvote_cv, z, patch_bbox, mip, inverse, serial, prefix,
               softmin_temp, blur_sigma):
    super().__init__(pairwise_cvs, vvote_cv, z, patch_bbox, mip, inverse, serial, prefix,
                     softmin_temp, blur_sigma)

  def execute(self, aligner):
    pairwise_cvs = {int(k): DCV(v) for k,v in self.pairwise_cvs.items()}
    vvote_cv = DCV(self.vvote_cv)
    z = self.z
    patch_bbox = deserialize_bbox(self.patch_bbox)
    mip = self.mip
    inverse = bool(self.inverse)
    serial = bool(self.serial)
    prefix = self.prefix
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
      with Storage(vvote_cv.path) as stor:
        path = 'vector_vote_done/{}/{}'.format(prefix, patch_bbox.stringify(z))
        stor.put_file(path, '')
        print('Marked finished at {}'.format(path))
      end = time()
      diff = end - start
      print('VectorVoteTask: {:.3f} s'.format(diff))


class CloudComposeTask(RegisteredTask):
  def __init__(self, f_cv, g_cv, dst_cv, f_z, g_z, dst_z, patch_bbox, f_mip, g_mip, 
                     dst_mip, factor, affine, pad, prefix):
    super().__init__(f_cv, g_cv, dst_cv, f_z, g_z, dst_z, patch_bbox, f_mip, g_mip, 
                     dst_mip, factor, affine, pad, prefix)

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
    prefix = self.prefix
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
      with Storage(dst_cv.path) as stor:
        path = 'compose_done/{}/{}'.format(prefix, patch_bbox.stringify(dst_z))
        stor.put_file(path, '')
        print('Marked finished at {}'.format(path))
      end = time()
      diff = end - start
      print('ComposeTask: {:.3f} s'.format(diff))


class CloudMultiComposeTask(RegisteredTask):
    def __init__(self, cv_list, dst_cv, z_list, dst_z, patch_bbox, mip_list,
                 dst_mip, factors, pad, prefix):
        super().__init__(cv_list, dst_cv, z_list, dst_z, patch_bbox, mip_list,
                         dst_mip, factors, pad, prefix)

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
        prefix = self.prefix
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
            with Storage(dst_cv.path) as stor:
                path = 'multi_compose_done/{}/{}'.format(
                    prefix, patch_bbox.stringify(dst_z))
                stor.put_file(path, '')
                print('Marked finished at {}'.format(path))
            end = time()
            diff = end - start
            print('MultiComposeTask: {:.3f} s'.format(diff))


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
  def __init__(self, model_path, src_cv, tgt_cv, z, tgt_range, patch_bbox, mip,
               w_cv, pad, softmin_temp, prefix):
    super().__init__(model_path, src_cv, tgt_cv, z, tgt_range, patch_bbox, mip,
               w_cv, pad, softmin_temp, prefix)

  def execute(self, aligner):
    patch_bbox = deserialize_bbox(self.patch_bbox)
    w_cv = DCV(self.w_cv)
    src_cv = DCV(self.src_cv)
    tgt_cv = DCV(self.tgt_cv)
    print("self tgt_range is", self.tgt_range)
    aligner.res_and_compose(self.model_path, src_cv, tgt_cv, self.z,
                            self.tgt_range, patch_bbox, self.mip, w_cv,
                            self.pad, self.softmin_temp)
    with Storage(w_cv.path) as stor:
      path = 'res_and_compose/{}-{}/{}'.format(self.prefix, self.mip,
                                               patch_bbox.stringify(self.z))
      stor.put_file(path, '')
      print('Marked finished at {}'.format(path))

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

class ComputeFcorrTask(RegisteredTask):
  def __init__(self, cv, dst_cv, dst_nopost, patch_bbox, mip, z1, z2, prefix):
    super(). __init__(cv, dst_cv, dst_nopost, patch_bbox, mip, z1, z2, prefix)

  def execute(self, aligner):
    cv = DCV(self.cv)
    dst_cv = DCV(self.dst_cv)
    dst_nopost = DCV(self.dst_nopost)
    z1 = self.z1
    z2 = self.z2
    patch_bbox = deserialize_bbox(self.patch_bbox)
    mip = self.mip
    print("\nFcorring "
          "cv {}\n"
          "z={} to z={}\n"
          "at MIP{}"
          "\n".format(cv, z1, z2, mip), flush=True)
    start = time()
    image, image_no = aligner.get_fcorr(patch_bbox, cv, mip, z1, z2)
    aligner.save_image(image, dst_cv, z2, patch_bbox, 8, to_uint8=False)
    aligner.save_image(image_no, dst_nopost, z2, patch_bbox, 8, to_uint8=False)
    with Storage(dst_cv.path) as stor:
      path = 'Fcorr_done/{}/{}'.format(self.prefix, patch_bbox.stringify(z2))
      stor.put_file(path, '')
      print('Marked finished at {}'.format(path))
    end = time()
    diff = end - start
    print('FcorrTask: {:.3f} s'.format(diff))
