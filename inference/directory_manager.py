from mipless_cloudvolume import MiplessCloudVolume as CV 
from copy import deepcopy, copy
from cloudvolume.lib import Vec
from os.path import join

class SrcDir():
  def __init__(self, src_path, tgt_path, 
                     src_mask_path, tgt_mask_path, 
                     src_mask_mip, tgt_mask_mip,
                     src_mask_val, tgt_mask_val):
    self.vols = {}
    self.kwargs = {'bounded': False, 'fill_missing': True, 'progress': False}
    self.vols['src_img'] = CV(src_path, mkdir=False, **self.kwargs) 
    self.vols['tgt_img'] = CV(tgt_path, mkdir=False, **self.kwargs) 
    if src_mask_path:
      self.vols['src_mask'] = CV(src_mask_path, mkdir=False, **self.kwargs) 
    if tgt_mask_path:
      self.vols['tgt_mask'] = CV(tgt_mask_path, mkdir=False, **self.kwargs) 
    self.src_mask_mip = src_mask_mip
    self.tgt_mask_mip = tgt_mask_mip
    self.src_mask_val = src_mask_val
    self.tgt_mask_val = tgt_mask_val

  def __getitem__(self, k):
    return self.vols[k]

  def __contains__(self, k):
    return k in self.vols

class DstDir():
  """Manager of CloudVolumes required by the Aligner
  
  Manage CloudVolumes used for reading & CloudVolumes used for writing. Read & write
  distinguished by the different sets of kwargs that are used for the CloudVolume.
  All CloudVolumes are MiplessCloudVolumes. 
  """
  def __init__(self, dst_path, info, provenance, suffix='', distributed=False):
    print('Creating DstDir for {0}'.format(dst_path))
    self.root = dst_path
    self.distributed = distributed
    self.info = info
    self.provenance = provenance
    self.paths = {} 
    self.dst_chunk_sizes = []
    self.dst_voxel_offsets = []
    self.vec_chunk_sizes = [] 
    self.vec_voxel_offsets = []
    self.vec_total_sizes = []
    self.compile_scales()
    self.read = {}
    self.write = {}
    #self.read_kwargs = {'bounded': False, 'fill_missing': True, 'progress': False}
    self.read_kwargs = {'bounded': False, 'progress': False}
    #self.write_kwargs = {'bounded': False, 'fill_missing': True, 'progress': False, 
    self.write_kwargs = {'bounded': False, 'progress': False, 
                  'autocrop': True, 'non_aligned_writes': False, 'cdn_cache': False}
    self.add_path('dst_img', join(self.root, 'image'), data_type='uint8', num_channels=1, fill_missing=True)
    self.add_path('dst_img_high_res', join(self.root, 'upsampled_image'), data_type='uint8', num_channels=1)
    self.add_path('field', join(self.root, 'field'), data_type='float32',
                  num_channels=2, fill_missing=True)
    self.suffix = suffix
    self.create_paths()
  
  def for_read(self, k):
    return self.read[k]

  def for_write(self, k):
    return self.write[k]
  
  def __getitem__(self, k):
    return self.read[k]

  def __contains__(self, k):
    return k in self.read

  @classmethod
  def create_info(cls, src_cv, mip_range, max_offset):
    src_info = src_cv.info
    m = len(src_info['scales'])
    each_factor = Vec(2,2,1)
    factor = Vec(2**m,2**m,1)
    max_mip = mip_range[-1]
    for k in range(m, max_mip+1):
      src_cv.add_scale(factor)
      factor *= each_factor
      chunksize = src_info['scales'][-2]['chunk_sizes'][0] // each_factor
      src_info['scales'][-1]['chunk_sizes'] = [ list(map(int, chunksize)) ]

    info = deepcopy(src_info)
    chunk_size = info["scales"][0]["chunk_sizes"][0][0]
    print("chunk_size is", chunk_size)
    dst_size_increase = max_offset
    if dst_size_increase % chunk_size != 0:
      dst_size_increase = dst_size_increase - (dst_size_increase % max_offset) + chunk_size
    scales = info["scales"]
    for i in range(len(scales)):
      scales[i]["voxel_offset"][0] -= int(dst_size_increase / (2**i))
      scales[i]["voxel_offset"][1] -= int(dst_size_increase / (2**i))

      scales[i]["size"][0] += int(dst_size_increase / (2**i))
      scales[i]["size"][1] += int(dst_size_increase / (2**i))

      x_remainder = scales[i]["size"][0] % scales[i]["chunk_sizes"][0][0]
      y_remainder = scales[i]["size"][1] % scales[i]["chunk_sizes"][0][1]

      x_delta = 0
      y_delta = 0
      if x_remainder != 0:
        x_delta = scales[i]["chunk_sizes"][0][0] - x_remainder
      if y_remainder != 0:
        y_delta = scales[i]["chunk_sizes"][0][1] - y_remainder

      scales[i]["size"][0] += x_delta
      scales[i]["size"][1] += y_delta

      scales[i]["size"][0] += int(dst_size_increase / (2**i))
      scales[i]["size"][1] += int(dst_size_increase / (2**i))
      #make it slice-by-slice writable
      scales[i]["chunk_sizes"][0][2] = 1
    return info

  @classmethod
  def create_info_batch(cls, src_cv, mip_range, max_offset, size_batch,
                       size_chunk, batch_mip):
    src_info = src_cv.info
    m = len(src_info['scales'])
    each_factor = Vec(2,2,1)
    factor = Vec(2**m,2**m,1)
    for _ in mip_range: 
      src_cv.add_scale(factor)
      factor *= each_factor
      chunksize = src_info['scales'][-2]['chunk_sizes'][0] // each_factor
      src_info['scales'][-1]['chunk_sizes'] = [ list(map(int, chunksize)) ]

    info = deepcopy(src_info)
    chunk_size = info["scales"][0]["chunk_sizes"][0][0]
    dst_size_increase = max_offset
    if dst_size_increase % chunk_size != 0:
      dst_size_increase = dst_size_increase - (dst_size_increase % max_offset) + chunk_size
    scales = info["scales"]
    scales[batch_mip]["chunk_sizes"][0][0] = size_chunk
    scales[batch_mip]["chunk_sizes"][0][1] = size_chunk
    for i in range(len(scales)):
      scales[i]["voxel_offset"][0] -= int(dst_size_increase / (2**i))
      scales[i]["voxel_offset"][1] -= int(dst_size_increase / (2**i))

      scales[i]["size"][0] += int(dst_size_increase / (2**i))
      scales[i]["size"][1] += int(dst_size_increase / (2**i))

      x_remainder = scales[i]["size"][0] % scales[i]["chunk_sizes"][0][0]
      y_remainder = scales[i]["size"][1] % scales[i]["chunk_sizes"][0][1]

      x_delta = 0
      y_delta = 0
      if x_remainder != 0:
        x_delta = scales[i]["chunk_sizes"][0][0] - x_remainder
      if y_remainder != 0:
        y_delta = scales[i]["chunk_sizes"][0][1] - y_remainder

      scales[i]["size"][0] += x_delta
      scales[i]["size"][1] += y_delta

      scales[i]["size"][0] += int(dst_size_increase / (2**i))
      scales[i]["size"][1] += int(dst_size_increase / (2**i))
      #make it slice-by-slice writable
      scales[i]["chunk_sizes"][0][2] = 1

    scales[batch_mip]["chunk_sizes"][0][2] = size_batch
    return info


  def compile_scales(self):
    scales = self.info["scales"]
    for i in range(len(scales)):
      self.dst_chunk_sizes.append(scales[i]["chunk_sizes"][0][0:2])
      self.dst_voxel_offsets.append(scales[i]["voxel_offset"]) 
      self.vec_chunk_sizes.append(scales[i]["chunk_sizes"][0][0:2])
      self.vec_voxel_offsets.append(scales[i]["voxel_offset"])
      self.vec_total_sizes.append(scales[i]["size"])

  def create_cv(self, k, ignore_info=False):
    path, data_type, channels, fill_missing = self.paths[k]
    provenance = self.provenance 
    info = deepcopy(self.info)
    info['data_type'] = data_type
    info['num_channels'] = channels
    if ignore_info:
      info = None
    self.read[k] = CV(path, mkdir=False, info=info, provenance=provenance, fill_missing=fill_missing, **self.read_kwargs)
    self.write[k] = CV(path, mkdir=not ignore_info, info=info, provenance=provenance, fill_missing=fill_missing, **self.write_kwargs)

  def add_path(self, k, path, data_type='uint8', num_channels=1, fill_missing=True):
    self.paths[k] = (path, data_type, num_channels, fill_missing)

  def create_paths(self):
    for k in self.paths.keys():
      self.create_cv(k)

  def get_composed_cv(self, compose_start, inverse, for_read):
    k = self.get_composed_key(compose_start, inverse)
    if for_read:
      return self.for_read(k)
    else:
      return self.for_write(k)

  def get_composed_key(self, compose_start, inverse):
    #k = 'vvote_F{0}'.format(self.suffix)
    k = 'vvote_F_{0}'.format(self.suffix)
    if inverse:
      k = 'vvote_invF_{0}'.format(self.suffix)
    #return '{0}_{1:04d}'.format(k, compose_start)
    return '{0}{1:04d}'.format(k, compose_start)

  def add_composed_cv(self, compose_start, inverse):
    """Create CloudVolume for storing composed vector fields

    Args
       compose_start: int, indicating the earliest section used for composing
       inverse: bool indicating whether composition aligns COMPOSE_START to Z (True),
        or Z to COMPOSE_START (False)
    """
    k = self.get_composed_key(compose_start, inverse)
    path = join(self.root, 'composed', self.get_composed_key(compose_start, inverse))
    self.add_path(k, path, data_type='float32', num_channels=2)
    self.create_cv(k)

