from cloudvolume import CloudVolume
from cloudvolume.lib import Bbox, Vec
import correspondences
import numpy as np

class Inspect():

  def __init__(self, cv, inspect_bbox, full_bbox, controller):
    self.cv = cv
    self.inspect_bbox = inspect_bbox
    self.full_bbox = full_bbox
    self.c = controller
    self.field = None
    self.pts = None
    self.field = self.load()
    self.pts = self.to_points(self.field)

  def load(self):
    return self.cv[self.inspect_bbox.to_slices() + (slice(0,2),)]

  # def save(self, field):
    # self.cv[self.bbox.to_slices()] = field

  def to_points(self, field):
    """Convert 4D field to list of points
    """
    pts = []
    inspect_offset = self.get_offset(self.inspect_bbox)
    full_offset = self.get_offset(self.full_bbox)
    scale = self.full_bbox.size3()[:2] / 2
    half_width = self.full_bbox.size3()
    half_width[2] = 0
    half_width //= 2
    for i in range(field.shape[0]):
      for j in range(field.shape[1]):
        d = field[i,j,0,::-1]
        d = np.multiply(d, scale)
        d = np.pad(d, (0,1), 'constant')
        src_pt = Vec(i,j,0) + inspect_offset
        dst_pt = d + full_offset + half_width
        pts.extend([self.scale_point(src_pt, 2**self.cv.mip), 
                    self.scale_point(dst_pt, 2**self.cv.mip)])
    return pts

  # def from_points(self, pts):
    # return fild

  def scale_point(self, pt, factor):
    return (int(round(pt[0] * factor)),
            int(round(pt[1] * factor)),
            int(round(pt[2])))

  def get_offset(self, bbox):
    return bbox.minpt
  
  def display(self):
    self.c.set(self.pts)
    # self.c.set_z(self.get_offset(self.inspect_bbox)[2])


class Field():
  """Display vector fields from CloudVolume in neuroglancer for inspection.
  """

  def __init__(self, cv_path, port=9999):
    self.controller = correspondences.controller(port)
    self.cv_path = cv_path

  def inspect(self, inspect_bbox, full_bbox, mip=6):
    cv = CloudVolume(self.cv_path, mip=mip, fill_missing=True)
    inspect_bbox = Bbox.from_slices(cv.slices_from_global_coords(inspect_bbox))
    full_bbox = Bbox.from_slices(cv.slices_from_global_coords(full_bbox))
    return Inspect(cv, inspect_bbox, full_bbox, self.controller)
