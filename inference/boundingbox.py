import numpy as np
import json

from util import crop

def deserialize_bbox(s):
  contents = json.loads(s)
  return BoundingBox(contents['m0_x'][0], contents['m0_x'][1],
                     contents['m0_y'][0], contents['m0_y'][1], mip=0, max_mip=contents['max_mip'])
class BoundingBox:
  def __init__(self, xs, xe, ys, ye, mip, max_mip=12):
    self.max_mip = max_mip
    scale_factor = 2**mip
    self.set_m0(xs*scale_factor, xe*scale_factor, ys*scale_factor, ye*scale_factor)

  def serialize(self):
    contents = {
      "max_mip": self.max_mip,
      "m0_x": self.m0_x,
      "m0_y": self.m0_y,
      "max_mip": self.max_mip
    }
    s = json.dumps(contents)
    return s


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
        import pdb; pdb.set_trace()
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
    norm = (full / (self.x_size(mip) -1)) * 2 - 1
    return norm

  def x_identity(self, mip):
    row  = np.arange(self.y_size(mip), dtype=np.float32)[:, np.newaxis]
    full = np.tile(row, (1, self.x_size(mip)))
    norm = (full / (self.y_size(mip)-1)) * 2 - 1
    return norm.T

  def identity(self, mip):
    x_id = self.x_identity(mip=mip)
    y_id = self.y_identity(mip=mip)
    result = np.stack((x_id, y_id), axis=2)
    return result

  def is_identity_flow(self, flow, mip):
    x_id = np.array_equal(self.x_identity(mip), flow[0, :, :, 0])
    y_id = np.array_equal(self.y_identity(mip), flow[0, :, :, 1])
    return x_id and y_id

  def x_res_displacement(self, d_pixels, mip):
    disp_prop = d_pixels / self.x_size(mip=0)
    result = np.full((self.x_size(mip), self.y_size(mip)), disp_prop, dtype=np.float32)
    return result

  def y_res_displacement(self, d_pixels, mip):
    disp_prop = d_pixels / self.y_size(mip=0)
    result = np.full((self.x_size(mip), self.y_size(mip)), disp_prop, dtype=np.float32)
    return result

  def spoof_x_y_residual(self, x_d, y_d, mip, crop_amount=0):
    x_res = crop(self.x_res_displacement(x_d, mip=mip), crop_amount)
    y_res = crop(self.y_res_displacement(y_d, mip=mip), crop_amount)
    result = np.stack((x_res, y_res), axis=2)
    result = np.expand_dims(result, 0)
    return result

  def __str__(self, mip):
    return "{}, {}".format(self.x_range(mip), self.y_range(mip))
