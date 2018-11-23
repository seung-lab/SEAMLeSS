from cloudvolume import CloudVolume, Storage

class MiplessCloudVolume():
  """Multi-mip access to CloudVolumes using the same path
  """
  def __init__(self, path, **kwargs):
    self.path = path
    self.mkdir = not self.exists()
    self.kwargs = kwargs
    self.cvs = {}

  def exists(self):
    s = Storage(self.path)
    return s.exists('info') 

  def create(self, mip):
    print('Creating CloudVolume for {0} at MIP{1}'.format(self.path, mip))
    self.cvs[mip] = CloudVolume(self.path, mip=mip, **self.kwargs)
    if self.mkdir:
      self.cvs[mip].commit_info()

  def __getitem__(self, mip):
    if mip not in self.cvs:
      self.create(mip)
    return self.cvs[mip]
 
  def __repr__(self):
    return self.path

