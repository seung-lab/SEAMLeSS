from mipless_cloudvolume import MiplessCloudVolume as CV 
from copy import deepcopy, copy
from cloudvolume.lib import Vec
from os.path import join

class CloudManager():
  """Manager of CloudVolumes required by the Aligner
  
  Manage CloudVolumes used for reading & CloudVolumes used for writing. Read & write
  distinguished by the different sets of kwargs that are used for the CloudVolume.
  All CloudVolumes are MiplessCloudVolumes. 
  """
  def __init__(self, info, provenance):
    self.info = info
    self.provenance = provenance
    self.num_scales = len(info['scales'])
    self.dst_chunk_sizes = []
    self.dst_voxel_offsets = []
    self.vec_chunk_sizes = [] 
    self.vec_voxel_offsets = []
    self.vec_total_sizes = []
    self.compile_scales()
    # self.read = {}
    # self.write = {}
    self.cvs = {}
    # self.read_kwargs = {'bounded': False, 'progress': False}
    self.kwargs = {'bounded': False, 'progress': False, 
                   'autocrop': True, 'non_aligned_writes': False, 
                   'cdn_cache': False}
  
  # def for_read(self, k):
  #   return self.read[k]

  # def for_write(self, k):
  #   return self.write[k]
  
  def __getitem__(self, k):
    return self.cvs[k]

  def __contains__(self, k):
    return k in self.cvs

  @classmethod
  def create_info(cls, src_cv, max_mip, max_offset, 
                       batch_size=-1, size_chunk=[-1,-1], batch_mip=-1):
    """Use an existing CloudVolume to make an info file for CloudVolumes to store
       outputs of the Aligner class.

    ASSUMPTIONS:
       The existing CloudVolume has MIP levels with x,y chunk_sizes that are factors
       of 1024 (set by the Aligner's chunk_size).

    Args:
       src_cv: existing CloudVolume object (must have an info file)
       max_mip: int for the highest MIP level required for Aligner outputs; adds
        those MIPs to the info file if they don't exist
       max_offset: int for the maximum padding required by the Aligner for its
        output objects; adjusts the voxel_offset of each MIP to account for this
        amount, then increases the size of each MIP to accommodate an integer number
        of chunk_sizes.
       batch_size: int for number of slices to include in adjusted MIP level's 
        chunk_size (optional)
       size_chunk: tuple for dimensions of adjusted MIP level's chunk_size (optional)
       batch_mip: int for MIP level where to adjust the chunk_size (optional)

    Returns:
       an info file that can accommodate the max_mip, max_offset
    """
    adjusting_chunksize = ((batch_size >= 0) and (all(i >= 0 for i in size_chunk)) and
                                                                       (batch_mip > 0))
    src_info = src_cv.info
    m = len(src_info['scales'])
    each_factor = Vec(2,2,1)
    factor = Vec(2**m,2**m,1)
    for k in range(m, max_mip+1):
      src_cv.add_scale(factor)
      factor *= each_factor
      chunksize = src_info['scales'][-2]['chunk_sizes'][0] // each_factor
      src_info['scales'][-1]['chunk_sizes'] = [ list(map(int, chunksize)) ]

    info = deepcopy(src_info)
    chunk_size = info["scales"][0]["chunk_sizes"][0][0]
    dst_size_increase = max_offset
    if dst_size_increase % chunk_size != 0:
      dst_size_increase = dst_size_increase + chunk_size
    scales = info["scales"]

    # adjust the chunk size for a given mip
    if adjusting_chunksize:
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

    if adjusting_chunksize: 
      scales[batch_mip]["chunk_sizes"][0][2] = batch_size 

    return info

  def compile_scales(self):
    """Compile the details about each scale in the info file for easier parsing
    """
    scales = self.info["scales"]
    for i in range(len(scales)):
      self.dst_chunk_sizes.append(scales[i]["chunk_sizes"][0][0:2])
      self.dst_voxel_offsets.append(scales[i]["voxel_offset"]) 
      self.vec_chunk_sizes.append(scales[i]["chunk_sizes"][0][0:2])
      self.vec_voxel_offsets.append(scales[i]["voxel_offset"])
      self.vec_total_sizes.append(scales[i]["size"])

  def create(self, path, data_type, num_channels, fill_missing, overwrite=False):
    """Create a MiplessCloudVolume based on params & details of class

    Args:
       path: str for path to CloudVolume
       data_type: str for data type to be used in info file for CloudVolume
       num_channels: int for number of channels used in info file for CloudVolume
       fill_missing: bool indicating whether to use fill_missing flag for the
         CloudVolume
       ignore_info: bool indicating whether to overwrite the info file for the
         CloudVolume

    Returns:
       read & write MiplessCloudVolumes
    """
    info = deepcopy(self.info)
    info['data_type'] = data_type
    info['num_channels'] = num_channels
    provenance = deepcopy(self.provenance)
    if not overwrite:
      print('Use existing info file for MiplessCloudVolume at {0}'.format(path))
      info = None
      provenance = None
    # self.read[path] = CV(path, mkdir=False, info=info, provenance=provenance, 
    #                   fill_missing=fill_missing, **self.read_kwargs)
    self.cvs[path] = CV(path, mkdir=overwrite, info=info, provenance=provenance, 
                       fill_missing=fill_missing, **self.kwargs)
    # return {'read': self.read[path], 'write': self.write[path]}
    return self.cvs[path]

