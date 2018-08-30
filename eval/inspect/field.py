from cloudvolume import CloudVolume
from cloudvolume.lib import Bbox, Vec
from correspondences import Controller
import numpy as np

from os.path import join

class Field():
  """Display vector fields from CloudVolume in neuroglancer for inspection.

    Args:
    * path: CloudVolume path to the root directory of a SEAMLeSS run
            images will be at path/image and vectors at path/vec
    * mip: mip level of the vectors to be displayed

    Example:
    ```
      from field import Field
      path = "gs://neuroglancer/seamless/matriarch_tile7_pinky100_pairs_write_res_v13"
      F = Field(path, mip=5)
      # open link provided, navigate to z=1345, & zoom in to an area of interest
      F.inspect(F.get_bbox())
    ```
  """
  def __init__(self, path, mip=5, vec_name='vec'):
    self.mip = mip
    self.root = path
    self.img_path = None
    self.vec_path = None
    self.cv = None
    self.controller = Controller()
    self.set_vector_path(join(path, vec_name))
    self.set_image_path(join(path, 'image'))

  def set_image_path(self, path):
    self.img_path = path
    layer = {'name': 'image',
              'url': join('precomputed://', self.img_path)}
    self.controller.set_layer(layer)
    print(self.controller.viewer)

  def set_vector_path(self, path):
    self.vec_path = path
    self.set_mip(self.mip)

  def set_mip(self, mip):
    self.mip = mip
    path = join(self.vec_path, str(mip))
    self.cv = CloudVolume(path, mip=mip, fill_missing=True)

  def display(self, pts):
    self.controller.set(pts)

  def get_bbox(self):
    return self.controller.get_bbox()

  def get_abs_residuals(self, bbox, bbox_mip=0):
    bbox_adj = Vec(2**(self.mip-bbox_mip), 2**(self.mip-bbox_mip), 1)
    mip_bbox = bbox // bbox_adj
    res = self.cv[mip_bbox.to_slices()][:,:,0,:]
    return res

  def inspect(self, bbox, bbox_mip=0):
    print('Inspecting {0}'.format(bbox))
    res = self.get_abs_residuals(bbox, bbox_mip)
    pts = self.to_points(res, bbox.minpt, self.mip)
    self.display(pts)
    # return pts

  def to_points(self, res, offset, mip):
    """Convert vec fields to list of points for neuroglancer
    """
    pts = []
    for n in range(res.shape[0]):
      for m in range(res.shape[1]):
        x0 = int(offset[0]) + n*(2**mip)
        y0 = int(offset[1]) + m*(2**mip)
        z0 = int(offset[2]-1)
        dx = int(np.round(res[n,m.0]))
        dy = int(np.round(res[n,m.1]))
        dz = 1
        pts.append([(x0, y0, z0), (x0+dx, y0+dy, z0+dz)])
    return pts