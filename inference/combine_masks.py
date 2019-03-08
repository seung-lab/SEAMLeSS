import numpy as np 
import torch
import torch.nn as nn

from cloudvolume import CloudVolume

from args import get_argparser, parse_args

from time import time

if __name__ == '__main__':
  parser = get_argparser()
  parser.add_argument('--src1_path', type=str)
  parser.add_argument('--src2_path', type=str)
  parser.add_argument('--src3_path', type=str)
  parser.add_argument('--dst_path', type=str)
  
  parser.add_argument('--src1_mip', type=int)
  parser.add_argument('--src2_mip', type=int)
  parser.add_argument('--src3_mip', type=int)
  parser.add_argument('--mip', type=int)

  args = parse_args(parser)

  src1_path = args.src1_path
  src2_path = args.src2_path
  src3_path = args.src3_path
  dst_path = args.dst_path  

  src1_mip = 4
  src2_mip = 6
  src3_mip = 8
  mip = 6
  section_range = np.arange(20000,22001)

  src1 = CloudVolume(src1_path, mip=src1_mip, parallel=True, progress=False)
  src2 = CloudVolume(src2_path, mip=src2_mip, parallel=True, progress=False)
  src3 = CloudVolume(src3_path, mip=src3_mip, parallel=True, progress=False)
  
  info = CloudVolume.create_new_info(
  	num_channels = 1,
  	layer_type = 'image',
  	data_type = 'uint8',
  	encoding = 'raw',
  	resolution = [2**(mip+2),2**(mip+2),40],
  	voxel_offset = [0,0,0],
  	chunk_size = [128,128,1],
  	volume_size = [9024,7424,28000]
  )

  dst = CloudVolume(dst_path, parallel=True, progress=False, cdn_cache=False, info=info)
  dst.commit_info()

  downsample = nn.AvgPool2d(kernel_size=(4,4), stride=(4,4), padding=0)
  upsample = nn.Upsample(scale_factor=(4,4), mode='bilinear')

  for i in section_range:
  	t0 = time()
  	print(">>>>>>> Section " + str(i))
  	s_src1 = torch.reshape(downsample(torch.reshape(torch.tensor(src1[0:36224,0:29824,i][:,:,0,0], dtype=torch.float32), (1,36224,29824))), (9056,7456))
  	s_src2 = torch.tensor(src2[0:9056,0:7456,i][:,:,0,0], dtype=torch.float32)
  	s_src3 = torch.reshape(upsample(torch.reshape(torch.tensor(src3[0:2256,0:1856,i][:,:,0,0], dtype=torch.float32), (1,1,2256,1856))), (9024,7424))

  	dst[:,:,i] = s_src1[:9024,:7424] + s_src2[:9024,:7424] + s_src3
  	print("Elapsed : " + str(np.round(time()-t0,3)))  


