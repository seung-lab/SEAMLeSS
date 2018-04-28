from process import Process
from cloudvolume import CloudVolume as cv
import numpy as np
import os
import json
from time import time
from copy import deepcopy

from util import crop, warp, upsample_flow, downsample_mip
from boundingbox import BoundingBox

from pathos.multiprocessing import ProcessPool, ThreadPool

class Aligner:
  def __init__(self, model_path, max_displacement, crop,
               mip_range, render_mip, high_mip_chunk,
               src_ng_path, dst_ng_path, is_Xmas=False, threads = 10,
               max_chunk = (1024, 1024), max_render_chunk = (2048*2, 2048*2)):

    self.high_mip       = mip_range[1]
    self.low_mip        = mip_range[0]
    self.render_mip     = render_mip
    self.high_mip_chunk = high_mip_chunk
    self.max_chunk      = max_chunk
    self.max_render_chunk = max_render_chunk

    self.max_displacement = max_displacement
    self.crop_amount = crop
    self.org_ng_path = src_ng_path
    self.src_ng_path = self.org_ng_path

    self.dst_ng_path = os.path.join(dst_ng_path, 'image')
    self.tmp_ng_path = os.path.join(dst_ng_path, 'intermediate')


    self.res_ng_paths  = [os.path.join(dst_ng_path, 'vec/{}'.format(i))
                                                    for i in range(self.high_mip + 10)] #TODO
    self.x_res_ng_paths = [os.path.join(r, 'x') for r in self.res_ng_paths]
    self.y_res_ng_paths = [os.path.join(r, 'y') for r in self.res_ng_paths]

    self.net = Process(model_path, is_Xmas=is_Xmas, cuda=True)

    self.dst_chunk_sizes   = []
    self.dst_voxel_offsets = []
    self.vec_chunk_sizes   = []
    self.vec_voxel_offsets = []
    self.vec_total_sizes   = []
    self._create_info_files(max_displacement)
    self.pool = ThreadPool(threads)

    #if not chunk_size[0] :
    #  raise Exception("The chunk size has to be aligned with ng chunk size")

  def set_chunk_size(self, chunk_size):
    self.high_mip_chunk = chunk_size

  def _create_info_files(self, max_offset):
    tmp_dir = "/tmp/{}".format(os.getpid())
    nocache_f = '"Cache-Control: no-cache"'

    os.system("mkdir {}".format(tmp_dir))

    src_info = cv(self.src_ng_path).info
    dst_info = deepcopy(src_info)

    ##########################################################
    #### Create dest info file
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

    cv(self.dst_ng_path, info=dst_info).commit_info()
    cv(self.tmp_ng_path, info=dst_info).commit_info()

    ##########################################################
    #### Create vec info file
    ##########################################################
    vec_info = deepcopy(src_info)
    vec_info["data_type"] = "float32"
    scales = deepcopy(vec_info["scales"])
    for i in range(len(scales)):
      self.vec_chunk_sizes.append(scales[i]["chunk_sizes"][0][0:2])
      self.vec_voxel_offsets.append(scales[i]["voxel_offset"])
      self.vec_total_sizes.append(scales[i]["size"])

      cv(self.x_res_ng_paths[i], info=vec_info).commit_info()
      cv(self.y_res_ng_paths[i], info=vec_info).commit_info()

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
                         mip=0, max_mip=self.high_mip)
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

    high_mip_scale = 2**(self.high_mip - mip)
    processing_chunk = (self.high_mip_chunk[0] * high_mip_scale,
                        self.high_mip_chunk[1] * high_mip_scale)
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


  ## Residual computation
  def run_net_test(self, s, t, mip):
    abs_residual = self.net.process(s, t, mip)

  def compute_residual_patch(self, source_z, target_z, out_patch_bbox, mip):
    #print ("Computing residual for {}".format(out_patch_bbox.__str__(mip=0)),
    #        end='', flush=True)
    precrop_patch_bbox = deepcopy(out_patch_bbox)
    precrop_patch_bbox.uncrop(self.crop_amount, mip=mip)

    if mip == self.high_mip:
      src_patch = self.get_image_data(self.src_ng_path, source_z, precrop_patch_bbox, mip)
    else:
      src_patch = self.get_image_data(self.tmp_ng_path, source_z, precrop_patch_bbox, mip)

    tgt_patch = self.get_image_data(self.dst_ng_path, target_z, precrop_patch_bbox, mip)

    abs_residual = self.net.process(src_patch, tgt_patch, mip, crop=self.crop_amount)
    #rel_residual = precrop_patch_bbox.spoof_x_y_residual(1024, 0, mip=mip,
    #                        crop_amount=self.crop_amount)
    self.save_residual_patch(abs_residual, source_z, out_patch_bbox, mip)


  def abs_to_rel_residual(self, abs_residual, patch, mip):
    x_fraction = patch.x_size(mip=0)
    y_fraction = patch.y_size(mip=0)

    rel_residual = deepcopy(abs_residual)
    rel_residual[0, :, :, 0] /= x_fraction
    rel_residual[0, :, :, 1] /= y_fraction
    return rel_residual


  ## Patch manipulation
  def warp_patch(self, ng_path, z, bbox, res_mip_range, mip):
    influence_bbox =  deepcopy(bbox)
    influence_bbox.uncrop(self.max_displacement, mip=0)

    agg_flow = influence_bbox.identity(mip=mip)
    agg_flow = np.expand_dims(agg_flow, axis=0)
    agg_res  = self.get_aggregate_rel_flow(z, influence_bbox, res_mip_range, mip)
    agg_flow += agg_res

    raw_data = self.get_image_data(ng_path, z, influence_bbox, mip)
    #no need to warp if flow is identity
    #warp introduces noise
    if not influence_bbox.is_identity_flow(agg_flow, mip=mip):
      warped   = warp(raw_data, agg_flow)
    else:
      #print ("not warping")
      warped = raw_data[0]

    mip_disp = int(self.max_displacement / 2**mip)
    result   = crop(warped, mip_disp)

    #preprocess divides by 256 and puts it into right dimensions
    #this data range is good already, so mult by 256
    return self.preprocess_data(result * 256)

  def downsample_patch(self, ng_path, z, bbox, mip):
    in_data = self.get_image_data(ng_path, z, bbox, mip - 1)
    result  = downsample_mip(in_data)
    return result

  ## Data saving
  def save_image_patch(self, ng_path, float_patch, z, bbox, mip):
    x_range = bbox.x_range(mip=mip)
    y_range = bbox.y_range(mip=mip)
    patch = float_patch[0, :, :, np.newaxis]
    uint_patch = (np.multiply(patch, 256)).astype(np.uint8)
    cv(ng_path, mip=mip, bounded=False, fill_missing=True, autocrop=True,
                                  progress=False)[x_range[0]:x_range[1],
                                                  y_range[0]:y_range[1], z] = uint_patch

  def save_residual_patch(self, flow, z, bbox, mip):
    #print ("Saving residual patch {} at mip {}".format(bbox.__str__(mip=0), mip), end='')
    start = time()
    self.save_vector_patch(flow, self.x_res_ng_paths[mip], self.y_res_ng_paths[mip], z, bbox, mip)
    end = time()
    #print (": {} sec".format(end - start))

  def save_vector_patch(self, flow, x_path, y_path, z, bbox, mip):
    x_res = flow[0, :, :, 0, np.newaxis]
    y_res = flow[0, :, :, 1, np.newaxis]

    x_range = bbox.x_range(mip=mip)
    y_range = bbox.y_range(mip=mip)

    cv(x_path, mip=mip, bounded=False, fill_missing=True, autocrop=True,
                                   progress=False)[x_range[0]:x_range[1],
                                                   y_range[0]:y_range[1], z] = x_res
    cv(y_path, mip=mip, bounded=False, fill_missing=True, autocrop=True,
                                   progress=False)[x_range[0]:x_range[1],
                                                   y_range[0]:y_range[1], z] = y_res

  ## Data loading
  def preprocess_data(self, data):
    sd = np.squeeze(data)
    ed = np.expand_dims(sd, 0)
    nd = np.divide(ed, float(256.0), dtype=np.float32)
    return nd

  def get_image_data(self, path, z, bbox, mip):
    x_range = bbox.x_range(mip=mip)
    y_range = bbox.y_range(mip=mip)

    data = cv(path, mip=mip, progress=False,
              bounded=False, fill_missing=True)[x_range[0]:x_range[1], y_range[0]:y_range[1], z]
    return self.preprocess_data(data)

  def get_vector_data(self, path, z, bbox, mip):
    x_range = bbox.x_range(mip=mip)
    y_range = bbox.y_range(mip=mip)

    data = cv(path, mip=mip, progress=False,
              bounded=False, fill_missing=True)[x_range[0]:x_range[1], y_range[0]:y_range[1], z]
    return data

  def get_abs_residual(self, z, bbox, mip):
    x = self.get_vector_data(self.x_res_ng_paths[mip], z, bbox, mip)[..., 0, 0]
    y = self.get_vector_data(self.y_res_ng_paths[mip], z, bbox, mip)[..., 0, 0]
    result = np.stack((x, y), axis=2)
    return np.expand_dims(result, axis=0)

  def get_rel_residual(self, z, bbox, mip):
    x = self.get_vector_data(self.x_res_ng_paths[mip], z, bbox, mip)[..., 0, 0]
    y = self.get_vector_data(self.y_res_ng_paths[mip], z, bbox, mip)[..., 0, 0]
    abs_res = np.stack((x, y), axis=2)
    abs_res = np.expand_dims(abs_res, axis=0)
    rel_res = self.abs_to_rel_residual(abs_res, bbox, mip)
    return rel_res


  def get_aggregate_rel_flow(self, z, bbox, res_mip_range, mip):
    result = np.zeros((1, bbox.x_size(mip), bbox.y_size(mip), 2), dtype=np.float32)
    start_mip = max(res_mip_range[0], self.low_mip)
    end_mip   = min(res_mip_range[1], self.high_mip)

    for res_mip in range(start_mip, end_mip + 1):
      scale_factor = 2**(res_mip - mip)

      rel_res = self.get_rel_residual(z, bbox, res_mip)
      up_rel_res = upsample_flow(rel_res, scale_factor)

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
    #for patch_bbox in chunks:
    def chunkwise(patch_bbox):
      #print ("Preparing future source {} at mip {}".format(patch_bbox.__str__(mip=0), mip),
      #        end='', flush=True)
      #start = time()
      warped_patch = self.warp_patch(self.src_ng_path, z, patch_bbox,
                                     (mip + 1, self.high_mip), mip)
      self.save_image_patch(self.tmp_ng_path, warped_patch, z, patch_bbox, mip)
    self.pool.map(chunkwise, chunks)
    end = time()
    print (": {} sec".format(end - start))

  def render(self, z, bbox, mip):
    print ("Rendering mip {}".format(mip),
              end='', flush=True)
    start = time()
    chunks = self.break_into_chunks(bbox, self.dst_chunk_sizes[mip],
                                    self.dst_voxel_offsets[mip], mip=mip, render=True)

    #for patch_bbox in chunks:
    def chunkwise(patch_bbox):
      #print ("Rendering {} at mip {}".format(patch_bbox.__str__(mip=0), mip),
      #          end='', flush=True)
      warped_patch = self.warp_patch(self.src_ng_path, z, patch_bbox,
                                    (mip, self.high_mip), mip)
      self.save_image_patch(self.dst_ng_path, warped_patch, z, patch_bbox, mip)
    self.pool.map(chunkwise, chunks)
    end = time()
    print (": {} sec".format(end - start))

  def render_section_all_mips(self, z, bbox):
    #total_bbox = self.get_upchunked_bbox(bbox, self.dst_chunk_sizes[self.high_mip],
    #                                           self.dst_voxel_offsets[self.high_mip],
    #                                           mip=self.high_mip)
    self.render(z, bbox, self.render_mip)
    self.downsample(z, bbox, self.render_mip, self.high_mip)

  def downsample(self, z, bbox, source_mip, target_mip):
    for m in range(source_mip+1, target_mip + 1):
      chunks = self.break_into_chunks(bbox, self.dst_chunk_sizes[m],
                                      self.dst_voxel_offsets[m], mip=m, render=True)

      #for patch_bbox in chunks:
      def chunkwise(patch_bbox):
        print ("Downsampling {} to mip {}".format(patch_bbox.__str__(mip=0), m))
        downsampled_patch = self.downsample_patch(self.dst_ng_path, z, patch_bbox, m)
        self.save_image_patch(self.dst_ng_path, downsampled_patch, z, patch_bbox, m)
      self.pool.map(chunkwise, chunks)

  def compute_section_pair_residuals(self, source_z, target_z, bbox):
    for m in range(self.high_mip,  self.low_mip - 1, -1):
      print ("Running net at mip {}".format(m),
                                end='', flush=True)
      start = time()
      chunks = self.break_into_chunks(bbox, self.vec_chunk_sizes[m],
                                      self.vec_voxel_offsets[m], mip=m)
      for patch_bbox in chunks:
      #def chunkwise(patch_bbox):
      #FIXME Torch runs out of memory
      #FIXME batchify download and upload
        self.compute_residual_patch(source_z, target_z, patch_bbox, mip=m)
      #self.pool.map(chunkwise, chunks)
      end = time()
      print (": {} sec".format(end - start))

      if m > self.low_mip:
          self.prepare_source(source_z, bbox, m - 1)



  ## Whole stack operations
  def align_ng_stack(self, start_section, end_section, bbox, move_anchor=True):
    if not self.check_all_params():
      raise Exception("Not all parameters are set")
    #if not bbox.is_chunk_aligned(self.dst_ng_path):
    #  raise Exception("Have to align a chunkaligned size")
    start = time()
    if move_anchor:
      for m in range(self.render_mip, self.high_mip + 1):
        self.copy_section(self.src_ng_path, self.dst_ng_path, start_section, bbox, mip=m)

    for z in range(start_section, end_section):
      self.compute_section_pair_residuals(z + 1, z, bbox)
      self.render_section_all_mips(z + 1, bbox)
    end = time()
    print ("Total time for aligning {} slices: {}".format(end_section - start_section,
                                                          end - start))
