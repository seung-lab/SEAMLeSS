from cloudvolume import CloudVolume, Storage
import json

def deserialize_miplessCV(s):
    contents = json.loads(s)
    return MiplessCloudVolume(contents['path'],mkdir = contents['mkdir'],
                              **contents['kwargs'])

class MiplessCloudVolume():
  """Multi-mip access to CloudVolumes using the same path
  """
  def __init__(self, path, mkdir, **kwargs):
    self.path = path
    self.mkdir = mkdir 
    self.kwargs = kwargs
    self.cvs = {}

  # def exists(self):
  #   s = Storage(self.path)
  #   return s.exists('info') 

  def serialize(self):
      contents = {
          "path" : self.path,
          "mkdir" : self.mkdir,
          "kwargs": self.kwargs,
      }
      s = json.dumps(contents)
      return s
  
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

