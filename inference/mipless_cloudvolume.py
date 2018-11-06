from cloudvolume import CloudVolume

class MiplessCloudVolume():
  """Multi-mip access to CloudVolumes using the same path
  """
  def __init__(self, path, mkdir=False, **kwargs):
    self.path = path
    self.mkdir = mkdir
    self.kwargs = kwargs
    self.cvs = {}

  def create(self, mip):
    self.cvs[mip] = CloudVolume(self.path, mip=mip, **self.kwargs)
    if self.mkdir:
      self.cvs[mip].commit_info()

  def __getitem__(self, mip):
    if mip not in self.cvs:
      self.create(mip)
    return self.cvs[mip]
 
  def __repr__(self):
    return self.path

