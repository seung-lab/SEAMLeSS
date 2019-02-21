from cloudvolume import CloudVolume, Storage
import json

def deserialize_miplessCV_old(s, cache={}):
    if s in cache:
      return cache[s]
    else:
      contents = json.loads(s)
      mcv = MiplessCloudVolume(contents['path'], mkdir=contents['mkdir'],
                               **contents['kwargs'])
      cache[s] = mcv
      return mcv

def deserialize_miplessCV_old2(s, cache={}):
    cv_kwargs = {'bounded': False, 'progress': False,
              'autocrop': False, 'non_aligned_writes': False,
              'cdn_cache': False}
    if s in cache:
      return cache[s]
    else:
      contents = json.loads(s)
      mcv = MiplessCloudVolume(contents['path'], mkdir=False,
                               fill_missing=True, **cv_kwargs)
      cache[s] = mcv
      return mcv

def deserialize_miplessCV(s, cache={}):
    cv_kwargs = {'bounded': False, 'progress': False,
              'autocrop': False, 'non_aligned_writes': False,
              'cdn_cache': False}
    if s in cache:
      return cache[s]
    else:
      mcv = MiplessCloudVolume(s, mkdir=False,
                               fill_missing=True, **cv_kwargs)
      cache[s] = mcv
      return mcv


class MiplessCloudVolume():
  """Multi-mip access to CloudVolumes using the same path
  """
  def __init__(self, path, mkdir, **kwargs):
    self.path = path
    self.mkdir = mkdir 
    self.kwargs = kwargs
    self.cvs = {}
    if self.mkdir:
        self.store_info()

  # def exists(self):
  #   s = Storage(self.path)
  #   return s.exists('info') 

  def serialize(self):
      contents = {
          "path" : self.path,
        #  "mkdir" : self.mkdir,
        #  "kwargs": self.kwargs,
      }
      s = json.dumps(contents)
      return s

  def store_info(self):
    tmp_cv = CloudVolume(self.path, **self.kwargs)
    tmp_cv.commit_info()
    tmp_cv.commit_provenance()

  def create(self, mip):
    print('Creating CloudVolume for {0} at MIP{1}'.format(self.path, mip))
    self.cvs[mip] = CloudVolume(self.path, mip=mip, **self.kwargs)
    #if self.mkdir:
    #  self.cvs[mip].commit_info()
    #  self.cvs[mip].commit_provenance()

  def __getitem__(self, mip):
    if mip not in self.cvs:
      self.create(mip)
    return self.cvs[mip]
 
  def __repr__(self):
    return self.path

