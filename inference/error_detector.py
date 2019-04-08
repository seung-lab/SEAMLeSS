import numpy as np
import torch
import torch.nn.functional as F


# Volume format convenient to extract patch.
class Volume():

  def __init__(self, A, patch_size, indexing='CENTRAL'):

    self.A = A
    self.patch_size = patch_size
    self.indexing = indexing

  def __getitem__(self, focus):

    A = self.A
    patch_size = self.patch_size

    if self.indexing == 'CENTRAL':
      corner = focus - np.array([x/2 for x in patch_size], dtype=np.int32)
      corner = np.reshape(corner,(-1,))

    elif self.indexing == 'CORNER':
      corner = focus

    else:
      raise Exception("Bad indexing scheme.")

    patch = A[:,:,corner[0]:corner[0]+patch_size[0],corner[1]:corner[1]+patch_size[1],corner[2]:corner[2]+patch_size[2]]

    return patch

  def __setitem__(self, focus, val):

  	patch_size = self.patch_size

  	if self.indexing == 'CENTRAL':
  		corner = focus - np.array([x/2 for x in patch_size], dtype=np.int32)
  		corner = np.reshape(corner, (-1,))

  	elif self.indexing == 'CORNER':
  		corner = focus

  	else:
  		raise Exception("Bad indexing scheme.")

  	self.A[:,:,corner[0]:corner[0]+patch_size[0],corner[1]:corner[1]+patch_size[1],corner[2]:corner[2]+patch_size[2]] = val


# Create binary object mask
def object_mask(img):

  shape = img.shape
  obj_id = img[tuple([shape[i]//2 for i in range(len(shape))])]
  
  if isinstance(img, (torch.Tensor)):    
    mask = torch.tensor(img==obj_id, dtype=torch.float32)
  else:
    mask = (img==obj_id).astype(np.float32)

  return mask


# Random coordinate generator.
def random_coord_valid(volume_size, patch_size, n=1):

  x = np.random.randint(low=patch_size[0]//2, high=volume_size[0]-patch_size[0]//2, size=n)
  y = np.random.randint(low=patch_size[1]//2, high=volume_size[1]-patch_size[1]//2, size=n)
  z = np.random.randint(low=patch_size[2]//2, high=volume_size[2]-patch_size[2]//2, size=n)

  x = np.reshape(x, [x.size,-1])
  y = np.reshape(y, [y.size,-1])
  z = np.reshape(z, [z.size,-1])

  random_coord = np.concatenate([x,y,z], axis=1)

  return random_coord


# Pack images into multichannel image.
def pack_inputs(obj, img):

	input_list = [obj, img]

	return torch.cat(input_list, dim=1)


# Create visited array
def visited_init(volume_size, patch_size):

	visited = np.zeros((1,1,)+tuple(patch_size), dtype='uint8')

	# Mark out edge
	visited[0,0,:patch_size[0]//2,:,:] = 1
	visited[0,0,:,patch_size[1]//2,:] = 1
	visited[0,0,:,:,patch_size[2]//2] = 1
	visited[0,0,volume_size[0]-patch_size[0]//2:,:,:]
	visited[0,0,:,volume_size[1]-patch_size[1]//2:,:]
	visited[0,0,:,:,volume_size[2]-patch_size[2]//2:]

	# Mark out boundaries
	visited[np.where(visited==0)] = 1

	return visited

# Inference chunk.
def inference(model, seg, img, patch_size):
	
	volume_size = seg.shape[2:]
	patch_size = patch_size[::-1]

	# Input volumes
	seg_vol = Volume(seg, patch_size)
	img_vol = Volume(img, patch_size)

	# Visited volume
	visited_patch_size = (16,160,160)
	visited = visited_init(volume_size, patch_size)
	vis_vol = Volume(visited, visited_patch_size)

	# Output volume
	error_map = np.zeros((1,1,)+tuple(patch_size), dtype='uint8')
	error_vol = Volume(error_map, patch_size)

	coverage = 0
	while coverage < 1:

		focus = random_coord_valid(volume_size, patch_size)

		seg_patch = seg_vol[focus]
		obj_patch = torch.tensor(object_mask(seg_patch))
		img_patch = torch.tensor(img_vol[focus])
		input_patch = pack_inputs(obj_patch.cuda(), img_patch.cuda())

		pred = torch.sigmoid(model(input_patch))
		pred_reshape = torch.reshape(pred, (1,)+pred.shape[2:]) 
		pred_upsample = F.interpolate(pred_reshape, scale_factor=16, mode="nearest").cpu().detach().numpy()
		error_vol[focus] = np.maximum(error_vol[focus], pred_upsample*obj_patch).astype('uint8')

		vis_vol[focus] = torch.from_numpy(vis_vol[focus]) + obj_patch[:,:,8:24,80:240,80:240]

		coverage = np.round(np.sum(vis_vol.A>=1)/np.prod(volume_size),4)
		print("Coverage = {}".format(coverage))

	return error_vol.A