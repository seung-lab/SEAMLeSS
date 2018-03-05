from process import Process
from cloudvolume import CloudVolume as cv
import numpy as np
import os
import json

from copy import deepcopy

class BoundingBox:
  def __init__(self, xs, xe, ys, ye):
    self.x = (xs, xe)
    self.y = (ys, ye)

  def uncrop(self, crop_xy):
    return BoundingBox(self.x[0] - crop_xy, self.x[1] + crop_xy,
            self.y[0] - crop_xy, self.y[1] + crop_xy)

class Aligner:
  def __init__(self, model_path, chunk_size, src_ng_path, dst_ng_path):
    self.chunk_size = chunk_size

    self.src_ng_path = src_ng_path
    self.dst_ng_path = os.path.join(dst_ng_path, 'image')
    self.vec_ng_path = os.path.join(dst_ng_path, 'vec')

    self.x_vec_ng_path = os.path.join(self.vec_ng_path, 'x')
    self.y_vec_ng_path = os.path.join(self.vec_ng_path, 'y')

    self.net = Process(model_path)
    self.create_info_files(1024)
    #if not chunk_size[0] :
    #  raise Exception("The chunk size has to be aligned with ng chunk size")

    self.crop_amount = 0

  def create_info_files(self, max_offset):
    tmp_dir = "/tmp/{}".format(os.getpid())
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

    with open(os.path.join(tmp_dir, "info.dst"), 'w') as f:
      json.dump(dst_info, f)
    os.system("gsutil cp {} {}".format(os.path.join(tmp_dir, "info.dst"),
                                       os.path.join(self.dst_ng_path, "image/info")))

    vec_info = deepcopy(src_info)
    vec_info["data_type"] = "float32"
    with open(os.path.join(tmp_dir, "info.vec"), 'w') as f:
      json.dump(vec_info, f)
    os.system("gsutil cp {} {}".format(os.path.join(tmp_dir, "info.vec"),
                                       os.path.join(self.dst_ng_path, "vec/x/info")))
    os.system("gsutil cp {} {}".format(os.path.join(tmp_dir, "info.vec"),
                                       os.path.join(self.dst_ng_path, "vec/y/info")))

    os.system("rm -rf {}".format(tmp_dir))

  def check_all_params(self):
    return True

  def align_ng_stack(self, start_section, end_section, mip, bbox):
    if not self.check_all_params():
      raise Exception("Not all parameters are set")
    #if not bbox.is_chunk_aligned(self.dst_ng_path):
    #  raise Exception("Have to align a chunkaligned size")

    '''allign sections one by one'''
    import pdb; pdb.set_trace()
    for z in range(start_section, end_section):
      self.align_ng_section_pair(z + 1, z, mip, bbox)

  def align_ng_section_pair(self, source_z, target_z, mip, bbox):
    '''allign a pair of sections from NG'''
    high_mip = mip
    low_mip = mip
    for m in range(high_mip,  low_mip - 1, -1):
      for xs in range(bbox.x[0], bbox.x[1], self.chunk_size[0]):
        for ys in range(bbox.y[0], bbox.y[1], self.chunk_size[1]):
          patch_bbox = BoundingBox(xs, xs + self.chunk_size[0], ys, ys + self.chunk_size[0])
          self.compute_flow_patch(source_z, target_z, m, patch_bbox)

  def compute_flow_patch(self, source_z, target_z, mip, out_patch_bbox):
    precrop_patch_bbox = out_patch_bbox.uncrop(self.crop_amount)

    src_patch = self.get_warped_src_patch(source_z, mip, precrop_patch_bbox)
    tgt_patch = self.get_patch(target_z, mip, precrop_patch_bbox)

    #mip2 corresponds to level0 in the net
    flow = self.net.process(src_patch, tgt_patch, mip - 2)
    cropped_flow = self.crop(flow, self.crop_amount)
    self.save_flow_patch(cropped_flow, source_z, mip, out_patch_bbox)

  def preprocess_data(self, data):
    sd = np.squeeze(data)
    ed = np.expand_dims(sd, 0)
    nd = np.divide(ed, float(256.0), dtype=np.float32)
    return nd

  def crop(self, d, c):
    return d

  def get_warped_src_patch(self, z, mip, bbox):
    data = cv(self.src_ng_path, mip=mip)[bbox.x[0]:bbox.x[1], bbox.y[0]:bbox.y[1], z]
    return self.preprocess_data(data)

  def get_patch(self, z, mip, bbox):
    data = cv(self.src_ng_path, mip=mip)[bbox.x[0]:bbox.x[1], bbox.y[0]:bbox.y[1], z]
    return self.preprocess_data(data)

  def save_flow_patch(self, flow, z, mip, bbox):
    import pdb; pdb.set_trace()
    x_flow = flow[0, :, :, 0]
    y_flow = flow[0, :, :, 1]

    cv(self.x_vec_ng_path, mip=mip)[bbox.x[0]:bbox.x[1], bbox.y[0]:bbox.y[1], z] = x_flow
    cv(self.y_vec_ng_path, mip=mip)[bbox.x[0]:bbox.x[1], bbox.y[0]:bbox.y[1], z] = y_flow

