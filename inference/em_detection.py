import numpy as np
import itertools

import torch


def chunk_bboxes(vol_size, chunk_size, overlap=(0,0)):

  x_bnds = bounds1D(vol_size[0], chunk_size[0], overlap[0])
  y_bnds = bounds1D(vol_size[1], chunk_size[1], overlap[1])

  bboxes = [tuple(zip(xs, ys))
            for (xs, ys) in itertools.product(x_bnds, y_bnds)]

  return bboxes

def bounds1D(full_width, step_size, overlap=0):

  assert step_size > 0, "invalid step_size: {}".format(step_size)
  assert full_width > 0, "invalid volume_width: {}".format(full_width)
  assert overlap >= 0, "invalid overlap: {}".format(overlap)

  start = 0
  end = step_size

  bounds = []
  while end < full_width:
    bounds.append((start, end))

    start += step_size - overlap
    end = start + step_size

  # last window
  end = full_width
  bounds.append((end-step_size, end))

  return bounds

def defect_detect(model, image, chunk_size, overlap):

  img_size = image.shape[2:]
  
  overlap = np.array(overlap)
  bboxes = chunk_bboxes(img_size, chunk_size, 2*overlap)

  pred = torch.zeros(image.shape)

  for b in bboxes:
    bs = b[0]
    be = b[1]
    xsize = be[0]-bs[0]
    ysize = be[1]-bs[1]

    patch = image[0,0,bs[0]:be[0],bs[1]:be[1]]

    if np.sum(patch)==0:
      pred_patch = torch.zeros((1,1,chunk_size[0],chunk_size[1])).cuda()

    else:
      patch = torch.reshape(patch,(1,1,xsize,ysize))
      pred_patch = model(patch)

    pred[0,0,bs[0]+overlap[0]:be[0]-overlap[0],bs[1]+overlap[1]:be[1]-overlap[1]] = pred_patch[0,0,overlap[0]:-overlap[0],overlap[1]:-overlap[1]]

  return pred
