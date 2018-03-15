from process import Process
from cloudvolume import CloudVolume as cv
from util import crop, warp, upsample
import numpy as np
import os
import json

from copy import deepcopy

class BoundingBox:
  def __init__(self, xs, xe, ys, ye, mip, max_mip=12):
    self.max_mip = max_mip
    scale_factor = 2**mip
    self.set_m0(xs*scale_factor, xe*scale_factor, ys*scale_factor, ye*scale_factor)

  def set_m0(self, xs, xe, ys, ye):
    self.m0_x = (int(xs), int(xe))
    self.m0_y = (int(ys), int(ye))
    self.m0_x_size = int(xe - xs)
    self.m0_y_size = int(ye - ys)

  def x_range(self, mip):
    assert(mip <= self.max_mip)
    scale_factor = 2**mip
    xs = int(self.m0_x[0] / scale_factor)
    xe = int(self.m0_x[1] / scale_factor)
    return (xs, xe)

  def y_range(self, mip):
    assert(mip <= self.max_mip)
    scale_factor = 2**mip
    ys = int(self.m0_y[0] / scale_factor)
    ye = int(self.m0_y[1] / scale_factor)
    return (ys, ye)

  def x_size(self, mip):
    assert(mip <= self.max_mip)
    scale_factor = 2**mip
    return int(self.m0_x_size / scale_factor)

  def y_size(self, mip):
    assert(mip <= self.max_mip)
    scale_factor = 2**mip
    return int(self.m0_y_size / scale_factor)

  def check_mips(self):
    for m in range(1, self.max_mip + 1):
      if self.m0_x_size % 2**m != 0:
        raise Exception('Bounding box problem at mip {}'.format(m))

  def crop(self, crop_xy, mip):
    scale_factor = 2**mip
    m0_crop_xy = crop_xy * scale_factor
    self.set_m0(self.m0_x[0] + m0_crop_xy,
                self.m0_x[1] - m0_crop_xy,
                self.m0_y[0] + m0_crop_xy,
                self.m0_y[1] - m0_crop_xy)
    self.check_mips()

  def uncrop(self, crop_xy, mip):
    scale_factor = 2**mip
    m0_crop_xy = crop_xy * scale_factor
    self.set_m0(self.m0_x[0] - m0_crop_xy,
                self.m0_x[1] + m0_crop_xy,
                self.m0_y[0] - m0_crop_xy,
                self.m0_y[1] + m0_crop_xy)
    self.check_mips()

  def zeros(self, mip):
    return np.zeros((self.x_size(mip), self.y_size(mip)), dtype=np.float32)

  def y_identity(self, mip):
    row  = np.arange(self.x_size(mip), dtype=np.float32)[:, np.newaxis]
    full = np.tile(row, (1, self.y_size(mip)))
    norm = (full / self.x_size(mip)) * 2 - 1
    return norm

  def x_identity(self, mip):
    row  = np.arange(self.y_size(mip), dtype=np.float32)[:, np.newaxis]
    full = np.tile(row, (1, self.x_size(mip)))
    norm = (full / self.y_size(mip)) * 2 - 1
    return norm.T

  def is_identity_flow(self, flow, mip):
    x_id = np.array_equal(self.x_identity(mip), flow[0, :, :, 0])
    y_id = np.array_equal(self.y_identity(mip), flow[0, :, :, 1])
    return x_id and y_id

  def __str__(self, mip):
    return "{}, {}".format(self.x_range(mip), self.y_range(mip))

class Aligner:
  def __init__(self, model_path, processing_chunk_size, max_displacement, crop,
               high_mip, low_mip, src_ng_path, dst_ng_path):
    self.processing_chunk_size = processing_chunk_size
    self.max_displacement = max_displacement
    self.crop_amount = crop

    self.src_ng_path = src_ng_path
    self.dst_ng_path = os.path.join(dst_ng_path, 'image')
    self.vec_ng_path = os.path.join(dst_ng_path, 'vec')

    self.x_vec_ng_path = os.path.join(self.vec_ng_path, 'x')
    self.y_vec_ng_path = os.path.join(self.vec_ng_path, 'y')

    self.net = Process(model_path)

    self.dst_chunk_sizes   = []
    self.dst_voxel_offsets = []
    self.vec_chunk_sizes   = []
    self.vec_voxel_offsets = []
    self._create_info_files(max_displacement)

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

    with open(os.path.join(tmp_dir, "info.dst"), 'w') as f:
      json.dump(dst_info, f)
    os.system("gsutil -h {} cp {} {}".format(nocache_f,
                                       os.path.join(tmp_dir, "info.dst"),
                                       os.path.join(self.dst_ng_path, "info")))

    vec_info = deepcopy(src_info)
    vec_info["data_type"] = "float32"
    scales = vec_info["scales"]
    for i in range(len(scales)):
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

      #make it slice-by-slice writable
      scales[i]["chunk_sizes"][0][2] = 1

      self.vec_chunk_sizes.append(scales[i]["chunk_sizes"][0][0:2])
      self.vec_voxel_offsets.append(scales[i]["voxel_offset"])

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

  def align_ng_stack(self, start_section, end_section, bbox):
    if not self.check_all_params():
      raise Exception("Not all parameters are set")
    #if not bbox.is_chunk_aligned(self.dst_ng_path):
    #  raise Exception("Have to align a chunkaligned size")
    self.produce_optical_flow(start_section, end_section, bbox)
    self.render_stack(start_section, end_section, bbox, mip=0)

  def produce_optical_flow(self, start_section, end_section, bbox):
    for z in range(start_section, end_section):
      self.compute_section_pair_residuals(z + 1, z, bbox)

  def compute_section_pair_residuals(self, source_z, target_z, bbox):
    for m in range(self.high_mip,  self.low_mip - 1, -1):
      raw_x_range = bbox.x_range(mip=m)
      raw_y_range = bbox.y_range(mip=m)

      x_chunk = self.vec_chunk_sizes[m][0]
      y_chunk = self.vec_chunk_sizes[m][1]

      x_offset = self.vec_voxel_offsets[m][0]
      y_offset = self.vec_voxel_offsets[m][1]

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

      #do the initial partial chunks to chunk align everything
      #corner
      if x_delta != 0 and y_delta != 0:
        patch_bbox = BoundingBox(x_start, x_start + x_chunk,
                                 y_start, y_start + y_chunk,
                                 mip=m, max_mip=self.high_mip)
        self.compute_residual_patch(source_z, target_z, patch_bbox, mip=m)

      #x seam
      if y_delta != 0:
        for xs in range(calign_x_range[0], calign_x_range[1], self.processing_chunk_size[0]):
          patch_bbox = BoundingBox(xs, xs + self.processing_chunk_size[0],
                                   y_start, y_start + y_chunk,
                                   mip=m, max_mip=self.high_mip)
          self.compute_residual_patch(source_z, target_z, patch_bbox, mip=m)
      #y seam
      if x_delta != 0:
        for ys in range(calign_y_range[0], calign_y_range[1], self.processing_chunk_size[0]):
          patch_bbox = BoundingBox(x_start, x_start + x_chunk,
                                   ys, ys + self.processing_chunk_size[0],
                                   mip=m, max_mip=self.high_mip)
          self.compute_residual_patch(source_z, target_z, patch_bbox, mip=m)

      #do the rest
      for xs in range(calign_x_range[0], calign_x_range[1], self.processing_chunk_size[0]):
        for ys in range(calign_y_range[0], calign_y_range[1], self.processing_chunk_size[1]):
          patch_bbox = BoundingBox(xs, xs + self.processing_chunk_size[0],
                                   ys, ys + self.processing_chunk_size[0],
                                   mip=m, max_mip=self.high_mip)
          self.compute_residual_patch(source_z, target_z, patch_bbox, mip=m)



  def compute_residual_patch(self, source_z, target_z, out_patch_bbox, mip):
    print ("Computing residual for {}".format(out_patch_bbox.__str__(mip=0)))
    precrop_patch_bbox = deepcopy(out_patch_bbox)
    precrop_patch_bbox.uncrop(self.crop_amount, mip=mip)

    src_patch = self.get_warped_patch(self.src_ng_path, source_z, precrop_patch_bbox, mip)
    tgt_patch = self.get_image_data(self.src_ng_path, target_z, precrop_patch_bbox, mip)

    residual = self.net.process(src_patch, tgt_patch, mip, crop=self.crop_amount)
    self.save_residual_patch(residual, source_z, out_patch_bbox, mip)

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

  def get_x_residual(self, z, bbox, mip):
    data = self.get_vector_data(self.x_vec_ng_path, z, bbox, mip)[..., 0, 0]
    return np.expand_dims(data, axis=0)

  def get_y_residual(self, z, bbox, mip):
    data = self.get_vector_data(self.y_vec_ng_path, z, bbox, mip)[..., 0, 0]
    return np.expand_dims(data, axis=0)


  def get_aggregate_flow(self, z, bbox, mip):
    x_result = bbox.x_identity(mip=mip)
    x_result = np.expand_dims(x_result, axis=0)
    y_result = bbox.y_identity(mip=mip)
    y_result = np.expand_dims(y_result, axis=0)

    start_mip = max(mip + 1, self.low_mip)
    for m in range(start_mip, self.high_mip + 1):
      scale_factor = 2**(m - mip)

      x_residual   = self.get_x_residual(z, bbox, m)
      y_residual   = self.get_y_residual(z, bbox, m)

      upsampled_x  = upsample(x_residual, scale_factor) * scale_factor
      upsampled_y  = upsample(y_residual, scale_factor) * scale_factor

      x_result    += upsampled_x
      y_result    += upsampled_y


    return np.stack((x_result, y_result), axis = 3)

  def get_warped_patch(self, ng_path, z, bbox, mip):
    influence_bbox =  deepcopy(bbox)
    influence_bbox.uncrop(self.max_displacement, mip=0)

    agg_flow = self.get_aggregate_flow(z, influence_bbox, mip)
    raw_data = self.get_image_data(ng_path, z, influence_bbox, mip)
    #no need to warp if flow is identity
    #warp introduces noise
    if not influence_bbox.is_identity_flow(agg_flow, mip=mip):
      warped   = warp(raw_data, agg_flow)
    else:
      print ("not warping")
      warped = raw_data[0]

    mip_disp = int(self.max_displacement / 2**mip)
    result   = crop(warped, mip_disp)

    #preprocess divides by 256 and puts it into right dimensions
    #this data range is good already, so mult by 256
    return self.preprocess_data(result * 256)

  def save_image_patch(self, float_patch, z, bbox, mip):
    print (bbox.__str__(mip=0))
    x_range = bbox.x_range(mip=mip)
    y_range = bbox.y_range(mip=mip)
    patch = float_patch[0, :, :, np.newaxis]
    uint_patch = (np.multiply(patch, 256)).astype(np.uint8)
    cv(self.dst_ng_path, mip=mip, bounded=False, fill_missing=True,
                                  progress=False)[x_range[0]:x_range[1],
                                                  y_range[0]:y_range[1], z] = uint_patch

  def save_residual_patch(self, residual, z, bbox, mip):
    x_res = residual[0, :, :, 0, np.newaxis]
    y_res = residual[0, :, :, 1, np.newaxis]

    x_range = bbox.x_range(mip=mip)
    y_range = bbox.y_range(mip=mip)

    cv(self.x_vec_ng_path, mip=mip, bounded=False, fill_missing=True,
                                   progress=False)[x_range[0]:x_range[1],
                                                   y_range[0]:y_range[1], z] = x_res
    cv(self.y_vec_ng_path, mip=mip, bounded=False, fill_missing=True,
                                   progress=False)[x_range[0]:x_range[1],
                                                   y_range[0]:y_range[1], z] = y_res

  def render(self, z, bbox, mip):
    '''x_range = bbox.x_range(mip=mip)
    y_range = bbox.y_range(mip=mip)

    for xs in range(x_range[0], x_range[1], self.processing_chunk_size[0]):
      for ys in range(y_range[0], y_range[1], self.processing_chunk_size[1]):
        patch_bbox = BoundingBox(xs, xs + self.processing_chunk_size[0],
                                 ys, ys + self.processing_chunk_size[0],
                                 mip=mip, max_mip=self.high_mip)
        warped_patch = self.get_warped_patch(self.src_ng_path, z, patch_bbox, mip)
        self.save_image_patch(warped_patch, z, patch_bbox, mip)
    '''
    raw_x_range = bbox.x_range(mip=mip)
    raw_y_range = bbox.y_range(mip=mip)

    x_chunk = self.dst_chunk_sizes[mip][0]
    y_chunk = self.dst_chunk_sizes[mip][1]

    x_offset = self.dst_voxel_offsets[mip][0]
    y_offset = self.dst_voxel_offsets[mip][1]

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

    #do the initial partial chunks to chunk align everything
    #corner
    if x_delta != 0 and y_delta != 0:
      patch_bbox = BoundingBox(x_start, x_start + x_chunk,
                               y_start, y_start + y_chunk,
                               mip=mip, max_mip=self.high_mip)
      warped_patch = self.get_warped_patch(self.src_ng_path, z, patch_bbox, mip)
      self.save_image_patch(warped_patch, z, patch_bbox, mip)

    #x seam
    if y_delta != 0:
      for xs in range(calign_x_range[0], calign_x_range[1], self.processing_chunk_size[0]):
        patch_bbox = BoundingBox(xs, xs + self.processing_chunk_size[0],
                                 y_start, y_start + y_chunk,
                                 mip=mip, max_mip=self.high_mip)
        warped_patch = self.get_warped_patch(self.src_ng_path, z, patch_bbox, mip)
        self.save_image_patch(warped_patch, z, patch_bbox, mip)
    #y seam
    if x_delta != 0:
      for ys in range(calign_y_range[0], calign_y_range[1], self.processing_chunk_size[0]):
        patch_bbox = BoundingBox(x_start, x_start + x_chunk,
                                 ys, ys + self.processing_chunk_size[0],
                                 mip=mip, max_mip=self.high_mip)
        warped_patch = self.get_warped_patch(self.src_ng_path, z, patch_bbox, mip)
        self.save_image_patch(warped_patch, z, patch_bbox, mip)

    #do the rest
    for xs in range(calign_x_range[0], calign_x_range[1], self.processing_chunk_size[0]):
      for ys in range(calign_y_range[0], calign_y_range[1], self.processing_chunk_size[1]):
        patch_bbox = BoundingBox(xs, xs + self.processing_chunk_size[0],
                                 ys, ys + self.processing_chunk_size[0],
                                 mip=mip, max_mip=self.high_mip)
        warped_patch = self.get_warped_patch(self.src_ng_path, z, patch_bbox, mip)
        self.save_image_patch(warped_patch, z, patch_bbox, mip)


