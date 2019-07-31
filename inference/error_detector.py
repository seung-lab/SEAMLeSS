import numpy as np
import torch
import torch.nn.functional as F

import itertools
import operator


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


# Valid random coordinate generator.
def random_coord_valid(volume_size, patch_size, n=1):

  x = np.random.randint(low=patch_size[0]//2, high=volume_size[0]-patch_size[0]//2, size=n)
  y = np.random.randint(low=patch_size[1]//2, high=volume_size[1]-patch_size[1]//2, size=n)
  z = np.random.randint(low=patch_size[2]//2, high=volume_size[2]-patch_size[2]//2, size=n)

  x = np.reshape(x, [x.size,-1])
  y = np.reshape(y, [y.size,-1])
  z = np.reshape(z, [z.size,-1])

  random_coord = np.concatenate([x,y,z], axis=1)

  return random_coord


# Random coordinate generator.
def random_coord(bbox_start, bbox_end, n=1):

	x = np.random.randint(low=bbox_start[0], high=bbox_end[0], size=n)
	y = np.random.randint(low=bbox_start[1], high=bbox_end[1], size=n)
	z = np.random.randint(low=bbox_start[2], high=bbox_end[2], size=n)

	x = np.reshape(x, (x.size,-1))
	y = np.reshape(y, (y.size,-1))
	z = np.reshape(z, (z.size,-1))

	coord = np.concatenate([x,y,z], axis=1)

	return coord


# Pack images into multichannel image.
def pack_inputs(obj, img):

	input_list = [obj, img]

	return torch.cat(input_list, dim=1)


# Create visited array
def visited_init(seg, volume_size, patch_size):

	visited = np.zeros((1,1,)+tuple(volume_size), dtype='uint8')

	# Mark out edge
	visited[0,0,:patch_size[0]//2,:,:] = 1
	visited[0,0,:,:patch_size[1]//2,:] = 1
	visited[0,0,:,:,:patch_size[2]//2] = 1
	visited[0,0,volume_size[0]-patch_size[0]//2:,:,:] = 1
	visited[0,0,:,volume_size[1]-patch_size[1]//2:,:] = 1
	visited[0,0,:,:,volume_size[2]-patch_size[2]//2:] = 1

	# Mark out boundaries
	visited[np.where(seg==0)] = 1

	return visited


# # Inference chunk.
# def inference(model, seg, img, patch_size):
	
# 	volume_size = seg.shape[2:]
# 	patch_size = patch_size[::-1]

# 	# Input volumes
# 	seg_vol = Volume(seg, patch_size)
# 	img_vol = Volume(img, patch_size)

# 	# Visited volume
# 	visited_patch_size = (16,80,80)
# 	visited = visited_init(seg, volume_size, patch_size)
# 	vis_vol = Volume(visited, visited_patch_size)

# 	# Output volume
# 	errormap = np.zeros((1,1,)+tuple(volume_size), dtype='float32')
# 	error_vol = Volume(errormap, patch_size)

# 	coverage = 0
# 	i = 0
# 	while coverage < 1:

# 		focus = random_coord_valid(volume_size, patch_size)[0]

# 		if vis_vol.A[0,0,focus[0],focus[1],focus[2]] >= 1:
# 			continue

# 		seg_patch = seg_vol[focus]
# 		obj_patch = torch.tensor(object_mask(seg_patch))
# 		img_patch = torch.tensor(img_vol[focus])
# 		input_patch = pack_inputs(img_patch.cuda(),obj_patch.cuda())
		
# 		pred = torch.sigmoid(model(input_patch))
# 		pred_upsample = F.interpolate(pred, scale_factor=(1,8,8), mode="nearest").cpu().detach()
# 		error_vol[focus] = np.maximum(error_vol[focus], pred_upsample*obj_patch)

# 		vis_vol[focus] = torch.from_numpy(vis_vol[focus]) + torch.tensor(obj_patch[:,:,8:24,40:120,40:120], dtype=torch.uint8)

# 		i = i + 1

# 		coverage = np.round(np.sum(vis_vol.A>=1)/np.prod(volume_size),2)
# 		if i % 100 == 0:
# 			print("Coverage = {}".format(coverage))
		
# 	return error_vol.A


def bounds1D(start, end, step_size, overlap=0):
  
  assert step_size > 0, "Invalid step_size: {}".format(step_size)
  assert end > start, "Invalid range: {} ~ {}".format(start, end)
  assert overlap >=0, "Invalid overlap: {}".format(overlap)

  s = start
  e = s + step_size

  bounds = []
  while e < end:
    
    bounds.append((s, e))

    s += step_size - overlap
    e = s + step_size

  e = end
  bounds.append((s,e))

  return bounds


def chunk_bboxes(bbox_start, bbox_end, chunk_size, overlap=(0,0,0)):

	x_bnds = bounds1D(bbox_start[0], bbox_end[0], chunk_size[0], overlap[0])
	y_bnds = bounds1D(bbox_start[1], bbox_end[1], chunk_size[1], overlap[1])
	z_bnds = bounds1D(bbox_start[2], bbox_end[2], chunk_size[2], overlap[2])

	bboxes = [tuple(zip(xs, ys, zs))
						for (xs, ys, zs) in itertools.product(x_bnds, y_bnds, z_bnds)]

	return bboxes


def sample_objects_chunked(vol_seg, volume_size, patch_size, visited_size, chunk_size, mip=0):

	seg = vol_seg.A

	mip_factor = 2**mip
	if mip > 0:
		
		seg = seg[:,:,:,::mip_factor,::mip_factor]
		volume_size = (volume_size[0],
										volume_size[1]//mip_factor,
										volume_size[2]//mip_factor)
		patch_size = (patch_size[0],
										patch_size[1]//mip_factor,
										patch_size[2]//mip_factor)
		visited_size = (visited_size[0],
										visited_size[1]//mip_factor,
										visited_size[2]//mip_factor)
		chunk_size = (chunk_size[0],
										chunk_size[1]//mip_factor,
										chunk_size[2]//mip_factor)

	visited = visited_init(seg, volume_size, patch_size)
	vol_visited = Volume(visited, visited_size)
	vol_seg = Volume(seg, patch_size)
	
	print("Sampling valid points...")

	focus_list = np.array([])
	
	bbox_start = [patch_size[i]//2 for i in range(3)]
	bbox_end = [volume_size[i]-patch_size[i]//2 for i in range(3)]

	bbox_chunks = chunk_bboxes(bbox_start, bbox_end, chunk_size)
	
	for bbox in bbox_chunks:

		bbox_start_chunk = bbox[0]
		bbox_end_chunk = bbox[1]
		
		print("\nSampling valid points for [{},{},{}] ~ [{},{},{}]".format(
			bbox_start_chunk[0], bbox_start_chunk[1], bbox_start_chunk[2],
			bbox_end_chunk[0], bbox_end_chunk[1], bbox_end_chunk[2]))

		cover = 0
		i = 0
		while cover < 1:

			focus = random_coord(bbox_start_chunk, bbox_end_chunk)[0]

			if vol_visited.A[0,0,focus[0],focus[1],focus[2]] >= 1:
				continue

			patch_seg = vol_seg[focus]
			patch_obj_mask = object_mask(patch_seg)
			patch_obj_mask_crop = patch_obj_mask[(0,0,)+tuple([
				slice(patch_size[i]//2-visited_size[i]//2, patch_size[i]//2+visited_size[i]//2)
				for i in range(len(patch_size))])]

			vol_visited[focus] = vol_visited[focus] + patch_obj_mask_crop

			n_covered = np.sum(vol_visited.A[(0,0)+tuple([slice(bbox_start_chunk[i], bbox_end_chunk[i])
																		for i in range(3)])]>=1)
			chunk_size = np.prod([bbox_end_chunk[i]-bbox_start_chunk[i] for i in range(3)])
			cover = np.round(n_covered/chunk_size,4)

			# Neglect dust pieces
			if np.sum(patch_obj_mask_crop) < 2000/(mip_factor**2):
				continue
			
			focus_list = np.concatenate((focus_list, focus))
			
			i += 1

			if i % 100 == 0 or i <= 10:
				print("{} covered.".format(cover))

	focus_list = np.reshape(focus_list,(-1,3)).astype(np.uint32)
	focus_list[:,1] = focus_list[:,1]*mip_factor
	focus_list[:,2] = focus_list[:,2]*mip_factor

	return focus_list 


def inference(model, seg, img, patch_size):
	
	volume_size = seg.shape[2:]
	patch_size = patch_size[::-1]
	chunk_size = (128,256,256)
	visited_patch_size = (16,80,80)

	# Input volumes
	seg_vol = Volume(seg, patch_size)
	img_vol = Volume(img, patch_size)

	# Output volume
	errormap = np.zeros((1,1,)+tuple(volume_size), dtype='float32')
	error_vol = Volume(errormap, patch_size)

	# Sample points
	focus_list = sample_objects_chunked(seg_vol, volume_size, patch_size, visited_patch_size, chunk_size, mip=2)
	
	n = focus_list.shape[0]
	i = 0
	for i in range(n):

		focus = focus_list[i,:]

		seg_patch = seg_vol[focus]
		obj_patch = torch.tensor(object_mask(seg_patch))
		img_patch = torch.tensor(img_vol[focus])
		input_patch = pack_inputs(img_patch.cuda(),obj_patch.cuda())
		
		pred = torch.sigmoid(model(input_patch))
		pred_upsample = F.interpolate(pred, scale_factor=(1,8,8), mode="nearest").cpu().detach()
		error_vol[focus] = np.maximum(error_vol[focus], pred_upsample*obj_patch)

		i = i + 1

		if i % 100 == 0:
			print("{} / {}".format(i,n))
		
	return error_vol.A