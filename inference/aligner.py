from process import Process
from cloudvolume import CloudVolume as cv
from util import crop, warp, upsample
import numpy as np
import os
import json

from copy import deepcopy

class BoundingBox:
  def __init__(self, xs, xe, ys, ye):
    self.x = (int(xs), int(xe))
    self.y = (int(ys), int(ye))
    self.x_size = int(xe - xs)
    self.y_size = int(ye - ys)

  def uncrop(self, crop_xy):
    return BoundingBox(self.x[0] - crop_xy, self.x[1] + crop_xy,
            self.y[0] - crop_xy, self.y[1] + crop_xy)

  def zeros(self):
    return np.zeros((self.x[1] - self.x[0], self.y[1] - self.y[0]), dtype=np.float32)

  def y_identity(self):
    row  = np.arange(self.x_size, dtype=np.float32)[:, np.newaxis]
    full = np.tile(row, (1, self.y_size))
    norm = (full / self.x_size) * 2 - 1
    return norm

  def x_identity(self):
    row  = np.arange(self.y_size, dtype=np.float32)[:, np.newaxis]
    full = np.tile(row, (1, self.x_size))
    norm = (full / self.y_size) * 2 - 1
    return norm.T

  def rescale(self, factor):
    return BoundingBox(self.x[0]*factor, self.x[1]*factor,
                       self.y[0]*factor, self.y[1]*factor)

class Aligner:
  def __init__(self, model_path, chunk_size, max_displacement,
               high_mip, low_mip, src_ng_path, dst_ng_path):
    self.chunk_size = chunk_size
    self.max_displacement = max_displacement
    self.crop_amount = 0

    self.src_ng_path = src_ng_path
    self.dst_ng_path = os.path.join(dst_ng_path, 'image')
    self.vec_ng_path = os.path.join(dst_ng_path, 'vec')

    self.x_vec_ng_path = os.path.join(self.vec_ng_path, 'x')
    self.y_vec_ng_path = os.path.join(self.vec_ng_path, 'y')

    self.net = Process(model_path)
    self._create_info_files(1024)

    self.high_mip = high_mip
    self.low_mip  = low_mip

    #if not chunk_size[0] :
    #  raise Exception("The chunk size has to be aligned with ng chunk size")


  def _create_info_files(self, max_offset):
    tmp_dir = "/tmp/{}".format(os.getpid())
    nocache_f = '"Cache-Control: no-cache"'

    os.system("mkdir {}".format(tmp_dir))
    os.system("gsutil cp {} {}".format(os.path.join(self.src_ng_path, "info"),
                                       os.path.join(tmp_dir, "info.src")))

    with open(os.path.join(tmp_dir, "info.src")) as f:
      src_info = json.load(f)
    dst_info = deepcopy(src_info)

    chunk_size = dst_info["scales"][0]["chunk_sizes"][0][0]
    dst_size_increase = max_offset
    if dst_size_increase % chunk_size != 0:
      dst_size_increase = dst_size_increase - (dst_size_increase % max_offset) + chunk_size
    scales = dst_info["scales"]
    for i in range(len(scales)):
      scales[i]["voxel_offset"][0] - dst_size_increase / (2**i)
      scales[i]["voxel_offset"][1] - dst_size_increase / (2**i)
      scales[i]["size"][0] + dst_size_increase / (2**i)
      scales[i]["size"][1] - dst_size_increase / (2**i)
      #make it slice-by-slice writable
      scales[i]["chunk_sizes"][0][2] = 1

    with open(os.path.join(tmp_dir, "info.dst"), 'w') as f:
      json.dump(dst_info, f)
    os.system("gsutil -h {} cp {} {}".format(nocache_f,
                                       os.path.join(tmp_dir, "info.dst"),
                                       os.path.join(self.dst_ng_path, "image/info")))

    vec_info = deepcopy(src_info)
    vec_info["data_type"] = "float32"
    scales = vec_info["scales"]
    for i in range(len(scales)):
      #make it slice-by-slice writable
      scales[i]["chunk_sizes"][0][2] = 1

    with open(os.path.join(tmp_dir, "info.vec"), 'w') as f:
      json.dump(vec_info, f)

    os.system("gsutil -h {} cp {} {}".format(nocache_f,
                                       os.path.join(tmp_dir, "info.vec"),
                                       os.path.join(self.x_vec_ng_path, "info")))
    os.system("gsutil -h {} cp {} {}".format(nocache_f,
                                       os.path.join(tmp_dir, "info.vec"),
                                       os.path.join(self.y_vec_ng_path, "info")))

    os.system("rm -rf {}".format(tmp_dir))

  def check_all_params(self):
    return True

  def align_ng_stack(self, start_section, end_section, mip, bbox):
    if not self.check_all_params():
      raise Exception("Not all parameters are set")
    #if not bbox.is_chunk_aligned(self.dst_ng_path):
    #  raise Exception("Have to align a chunkaligned size")

    '''allign sections one by one'''
    for z in range(start_section, end_section):
      self.align_ng_section_pair(z + 1, z, mip, bbox)

  def align_ng_section_pair(self, source_z, target_z, mip, bbox):
    '''allign a pair of sections from NG'''
    for m in range(self.high_mip,  self.low_mip - 1, -1):
      mip_bbox = bbox.rescale(2**(mip - m))

      for xs in range(mip_bbox.x[0], mip_bbox.x[1], self.chunk_size[0]):
        for ys in range(mip_bbox.y[0], mip_bbox.y[1], self.chunk_size[1]):
          patch_bbox = BoundingBox(xs, xs + self.chunk_size[0], ys, ys + self.chunk_size[0])
          self.compute_flow_patch(source_z, target_z, m, patch_bbox)

    self.render_section(source_z, 0, bbox)

  def compute_residual_patch(self, source_z, target_z, mip, out_patch_bbox):
    precrop_patch_bbox = out_patch_bbox.uncrop(self.crop_amount)

    src_patch = self.get_warped_patch(self.src_ng_path, source_z, mip, precrop_patch_bbox)
    tgt_patch = self.get_image_data(self.src_ng_path, target_z, mip, precrop_patch_bbox)

    #mip2 corresponds to level0 in the net
    residual = self.net.process(src_patch, tgt_patch, mip - 2, crop=self.crop_amount)
    self.save_residual_patch(residual, source_z, mip, out_patch_bbox)

  def preprocess_data(self, data):
    sd = np.squeeze(data)
    ed = np.expand_dims(sd, 0)
    nd = np.divide(ed, float(256.0), dtype=np.float32)
    return nd

  def get_image_data(self, path, z, mip, bbox):
    data = cv(path, mip=mip,
              bounded=False, fill_missing=True)[bbox.x[0]:bbox.x[1], bbox.y[0]:bbox.y[1], z]
    return self.preprocess_data(data)

  def get_vector_data(self, path, z, mip, bbox):
    data = cv(path, mip=mip,
           bounded=False, fill_missing=True)[bbox.x[0]:bbox.x[1], bbox.y[0]:bbox.y[1], z]
    return data

  def get_x_residual(self, z, mip, bbox):
    data = self.get_vector_data(self.x_vec_ng_path, z, mip, bbox)[..., 0, 0]
    return np.expand_dims(data, axis=0)

  def get_y_residual(self, z, mip, bbox):
    data = self.get_vector_data(self.y_vec_ng_path, z, mip, bbox)[..., 0, 0]
    return np.expand_dims(data, axis=0)


  def get_aggregate_flow(self, z, mip, bbox):
    x_result = bbox.x_identity()
    x_result = np.expand_dims(x_result, axis=0)
    y_result = bbox.y_identity()
    y_result = np.expand_dims(y_result, axis=0)

    start_mip = min(mip + 1, self.low_mip)
    for m in range(start_mip, self.high_mip + 1):
      scale_factor = 2**(m - mip)
      m_bbox = bbox.rescale(1.0 / scale_factor)

      x_residual   = self.get_x_residual(z, m, m_bbox)
      y_residual   = self.get_y_residual(z, m, m_bbox)

      upsampled_x  = upsample(x_residual, scale_factor) * scale_factor
      upsampled_y  = upsample(y_residual, scale_factor) * scale_factor

      x_result    += upsampled_x
      y_result    += upsampled_y


    return np.stack((x_result, y_result), axis = 3)

  def get_warped_patch(self, ng_path, z, mip, bbox):
    mip_disp = int(self.max_displacement / (2**mip))
    influence_bbox =  bbox.uncrop(mip_disp)

    agg_flow = self.get_aggregate_flow(z, mip, influence_bbox)

    raw_data = self.get_image_data(ng_path, z, mip, influence_bbox)
    warped   = warp(raw_data, agg_flow)
    result   = crop(warped, mip_disp)

    return self.preprocess_data(result)

  def save_residual_patch(self, residual, z, mip, bbox):
    x_res = residual[0, :, :, 0, np.newaxis]
    y_res = residual[0, :, :, 1, np.newaxis]

    cv(self.x_vec_ng_path, mip=mip, bounded=False)[bbox.x[0]:bbox.x[1], bbox.y[0]:bbox.y[1], z] = x_res
    cv(self.y_vec_ng_path, mip=mip, bounded=False)[bbox.x[0]:bbox.x[1], bbox.y[0]:bbox.y[1], z] = y_res

  '''def render_section(self, z, mip, bbox):
    for xs in range(mip_bbox.x[0], mip_bbox.x[1], self.chunk_size[0]):
      for ys in range(mip_bbox.y[0], mip_bbox.y[1], self.chunk_size[1]):
        patch_bbox = BoundingBox(xs, xs + self.chunk_size[0], ys, ys + self.chunk_size[0])
        warped = self.get_warped_patch(self.src_ng_path, z, mip, patch_bbox)'''

