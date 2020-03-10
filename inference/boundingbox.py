import numpy as np
from math import floor, ceil
from utilities.helpers import crop
from cloudvolume.lib import Bbox
import json

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

  def contains(self, other):
    assert type(other) == type(self)

    dxs = other.m0_x[0] >= self.m0_x[0]
    dys = other.m0_y[0] >= self.m0_y[0]
    dxe = self.m0_x[1] >= other.m0_x[1]
    dye = self.m0_y[1] >= other.m0_y[1]

    return dxs and dys and dxe and dye

  def get_bounding_pts(self):
    return (self.m0_x[0], self.m0_y[0]), (self.m0_x[1], self.m0_y[1])

  def intersects(self, other):
    assert type(other) == type(self)

    if other.m0_x[1] < self.m0_x[0]:
      return False

    if other.m0_y[1] < self.m0_y[0]:
      return False

    if self.m0_x[1] < other.m0_x[0]:
      return False

    if self.m0_y[1] < other.m0_y[0]:
      return False

    return True

  def insets(self, other, mip):
    assert type(other) == type(self)
    assert mip <= self.max_mip

    xs, xe = self.x_range(mip)
    ys, ye = self.y_range(mip)
    oxs, oxe = other.x_range(mip)
    oys, oye = other.y_range(mip)

    return max(xs - oxs, 0), max(ys - oys, 0), xe-xs, ye-ys
    
  def set_m0(self, xs, xe, ys, ye):
    self.m0_x = (int(xs), int(xe))
    self.m0_y = (int(ys), int(ye))
    self.m0_x_size = int(xe - xs)
    self.m0_y_size = int(ye - ys)
 
  def get_offset(self, mip=0):
    scale_factor = 2**mip
    return (self.m0_x[0] / scale_factor + self.m0_x_size / 2 / scale_factor,
            self.m0_y[0] / scale_factor + self.m0_y_size / 2 / scale_factor)

  def x_range(self, mip):
    assert(mip <= self.max_mip)
    scale_factor = 2**mip
    xs = floor(self.m0_x[0] / scale_factor)
    xe = ceil(self.m0_x[1] / scale_factor)
    return (xs, xe)

  def y_range(self, mip):
    assert(mip <= self.max_mip)
    scale_factor = 2**mip
    ys = floor(self.m0_y[0] / scale_factor)
    ye = ceil(self.m0_y[1] / scale_factor)
    return (ys, ye)

  def x_size(self, mip):
    assert(mip <= self.max_mip)
    x_range = self.x_range(mip)
    return int(x_range[1] - x_range[0])

  def y_size(self, mip):
    assert(mip <= self.max_mip)
    y_range = self.y_range(mip)
    return int(y_range[1] - y_range[0])

  def check_mips(self):
    for m in range(1, self.max_mip + 1):
      if self.m0_x_size % 2**m != 0:
        # import pdb; pdb.set_trace()
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
    """Uncrop the bounding box by crop_xy at given MIP level
    """
    scale_factor = 2**mip
    m0_crop_xy = crop_xy * scale_factor
    self.set_m0(self.m0_x[0] - m0_crop_xy,
                self.m0_x[1] + m0_crop_xy,
                self.m0_y[0] - m0_crop_xy,
                self.m0_y[1] + m0_crop_xy)
    self.check_mips()

  def zeros(self, mip):
    return np.zeros((self.x_size(mip), self.y_size(mip)), dtype=np.float32)

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

  def __str__(self, mip=0):
    return "{}, {}".format(self.x_range(mip), self.y_range(mip))
  
  def __repr__(self):
    return self.__str__(mip=0)
  
  @classmethod
  def from_bbox(cls, bbox, mip, max_mip):
    """Create BoundingBox from CloudVolume.lib Bbox
    """
    return cls(xs=bbox.minpt[0], 
               xe=bbox.maxpt[0], 
               ys=bbox.minpt[1], 
               ye=bbox.maxpt[1], 
               mip=mip,
               max_mip=max_mip)
  
  def to_bbox(self, z_start, z_stop=None, mip=0):
    x_start = self.x_range(mip)[0]    
    x_stop = self.x_range(mip)[1]    
    y_start = self.y_range(mip)[0]    
    y_stop = self.y_range(mip)[1]    
    if not z_stop:
      z_stop = z_start+ 1
    return Bbox([x_start, y_start, z_start],
                [x_stop, y_stop, z_stop])

  def stringify(self, z_start, z_stop=None, mip=0):
    x_start = self.x_range(mip)[0]    
    x_stop = self.x_range(mip)[1]    
    y_start = self.y_range(mip)[0]    
    y_stop = self.y_range(mip)[1]    
    if not z_stop:
      z_stop = z_start+ 1
    return '{0},{1},{2}_{3},{4},{5}'.format(x_start, y_start, z_start, 
                                            x_stop, y_stop, z_stop)
    
