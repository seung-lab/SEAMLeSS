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

# def defect_detect(model, image, chunk_size, overlap):

#   img_size = image.shape[2:]
  
#   overlap = np.array(overlap)
#   bboxes = chunk_bboxes(img_size, chunk_size, 2*overlap)

#   pred = torch.zeros(image.shape)

#   for b in bboxes:
#     bs = b[0]
#     be = b[1]
#     xsize = be[0]-bs[0]
#     ysize = be[1]-bs[1]

#     patch = image[0,0,bs[0]:be[0],bs[1]:be[1]]

#     patch = torch.reshape(patch,(1,1,xsize,ysize))
#     pred_patch = model(patch)

#     pred[0,0,bs[0]+overlap[0]:be[0]-overlap[0],bs[1]+overlap[1]:be[1]-overlap[1]] = pred_patch[0,0,overlap[0]:-overlap[0],overlap[1]:-overlap[1]]

#   return pred

def defect_detect(model, image, chunk_size, overlap):

  img_size = image.shape[2:]
  
  overlap = np.array(overlap)
  bboxes = chunk_bboxes(img_size, chunk_size, 2*overlap)

  # pred = torch.zeros(image.shape).cuda()
  pred = torch.zeros(image.shape)

  for b in bboxes:
    bs = b[0]
    be = b[1]

    xs = bs[0]; xe = be[0]; ys = bs[1]; ye = be[1];
    patch = image[0,0,xs:xe,ys:ye]

    img = patch.cpu().numpy().reshape((chunk_size[0],chunk_size[1]))

    if np.sum(img)==0:
      # pred_patch = torch.zeros((1,1,chunk_size[0],chunk_size[1])).cuda()
      pred_patch = torch.zeros((1,1,chunk_size[0],chunk_size[1]))

    else:
      img_rowl = np.cumsum(np.sum(img,axis=1))
      img_colu = np.cumsum(np.sum(img,axis=0))
      img_rowr = np.cumsum(np.sum(img,axis=1)[::-1])
      img_colb = np.cumsum(np.sum(img,axis=0)[::-1])

      idxl = np.where(img_rowl==0)[0]
      idxu = np.where(img_colu==0)[0]
      idxr = np.where(img_rowr==0)[0]
      idxb = np.where(img_colb==0)[0]
      
      if idxl.shape[0] or idxu.shape[0] or idxr.shape[0] or idxb.shape[0]:
        if idxl.shape[0]:
          xs = bs[0]+idxl[-1]-overlap[1]
          # patch = image[0,0,xs:xs+chunk_size[0],ys:ye]
        if idxu.shape[0]:
          ys = bs[1]+idxu[-1]-overlap[1]
          # patch = image[0,0,xs:xe,ys:ys+chunk_size[1]]
        if idxr.shape[0]:
          xs = be[0]-idxr[-1]-chunk_size[0]+overlap[0]
        if idxb.shape[0]:
          ys = be[1]-idxb[-1]-chunk_size[1]+overlap[1]
          
        if (xs+chunk_size[0])>=img_size[0]:
          xs = img_size[0]-chunk_size[0]
        elif xs<0:
          xs = 0
        if (ys+chunk_size[1])>=img_size[1]:
          ys = img_size[1]-chunk_size[1]
        elif ys<0:
          ys = 0
  
        patch = image[0,0,xs:xs+chunk_size[0],ys:ys+chunk_size[1]]
     
      patch = torch.reshape(patch,(1,1,chunk_size[0],chunk_size[1]))
      pred_patch = model(patch)

    pred[0,0,xs+overlap[0]:xs+chunk_size[0]-overlap[0],ys+overlap[1]:ys+chunk_size[1]-overlap[1]] += pred_patch[0,0,overlap[0]:-overlap[0],overlap[1]:-overlap[1]]

  return pred 


