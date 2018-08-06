from cloudvolume import CloudVolume
from cloudvolume.lib import Bbox, Vec
import correspondences
import numpy as np

from os.path import join

class Field():
  """Display vector fields from CloudVolume in neuroglancer for inspection.
  """
  def __init__(self, vec_path, mip=5, port=9999):
    self.controller = correspondences.controller(port)
    self.vec_path = vec_path
    self.mip = mip
    self.x_cv = CloudVolume(join(self.vec_path, str(mip), 'x'), mip=mip, fill_missing=True)
    self.y_cv = CloudVolume(join(self.vec_path, str(mip), 'y'), mip=mip, fill_missing=True)

  def display(self, pts):
    self.controller.set(pts)

  def get_abs_residuals(self, bbox, bbox_mip=0):
    bbox_adj = Vec(2**(self.mip-bbox_mip), 2**(self.mip-bbox_mip), 1)
    mip_bbox = bbox // bbox_adj
    x = self.x_cv[mip_bbox.to_slices()]
    y = self.y_cv[mip_bbox.to_slices()]
    return x, y

  def inspect(self, bbox, bbox_mip=0):
    print('Inspecting {0}'.format(bbox))
    x, y = self.get_abs_residuals(bbox, bbox_mip)
    pts = self.to_points(x, y, bbox.minpt, self.mip)
    self.display(pts)
    return pts

  def to_points(self, x, y, offset, mip):
    """Convert Neuroflow vec fields to list of points for neuroglancer
    """
    pts = []
    for n in range(x.shape[0]):
      for m in range(x.shape[1]):
        x0 = int(offset[0]) + n*(2**mip)
        y0 = int(offset[1]) + m*(2**mip)
        z0 = int(offset[2]-1)
        dx = int(np.round(y[n,m]))
        dy = int(np.round(x[n,m]))
        dz = 1
        pts.extend([(x0, y0, z0), (x0+dx, y0+dy, z0+dz)])
    return pts

# from field import Field
# F = Field("gs://neuroglancer/seamless/matriarch_tile7_drosophila_pairs_v0/vec", mip=5)
# F.inspect(F.controller.get_bbox())