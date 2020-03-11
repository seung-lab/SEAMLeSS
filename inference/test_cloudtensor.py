import pytest

from boundingbox import BoundingBox, BoundingCube
import numpy as np
import torch
from fields import Field
from cloudtensor import CloudTensor, CloudField
from cloudtensor import MiplessCloudTensor, MiplessCloudField

import shutil
import os

def delete_layer(path):
    if os.path.exists(path):
        shutil.rmtree(path)  

def test_cloudtensor():
    sz = [16, 16, 1]
    info = CloudTensor.create_new_info(
             num_channels=1, 
             layer_type='image', 
             data_type='uint8', 
             encoding='raw',
             resolution=[ 1,1,1 ], 
             voxel_offset=[0,0,0], 
             volume_size=sz,
             chunk_size=sz,
             )
    path = 'file:///tmp/cloudvolume/empty_volume'
    vol = CloudTensor(path, device='cpu', mip=0, info=info)
    assert(isinstance(vol, CloudTensor))
    cv = vol.cv
    vol = CloudTensor.from_cv(cv=cv, device='cpu')
    assert(isinstance(vol, CloudTensor))
    vol.commit_info()
    assert(vol.dtype == 'uint8')
    # create test image
    sz = 16
    bbox = BoundingBox(xs=0, xe=16, ys=0, ye=16, mip=0, max_mip=4)
    data = np.ones((1, 1, sz, sz), dtype=np.float32)
    f = torch.from_numpy(data)
    bcube = BoundingCube.from_bbox(bbox, zs=0, ze=1)
    vol[bcube] = f 
    g = vol[bcube]
    assert(torch.equal(f, g))
    delete_layer('/tmp/cloudvolume/empty_volume')

def test_cloudfield():
    sz = [16, 16, 1]
    info = CloudField.create_new_info(
             num_channels=2, 
             layer_type='image', 
             data_type='int16', 
             encoding='raw',
             resolution=[ 1,1,1 ], 
             voxel_offset=[0,0,0], 
             volume_size=sz,
             chunk_size=sz,
             )
    path = 'file:///tmp/cloudvolume/empty_volume'
    vol = CloudField(path, device='cpu', mip=0, info=info)
    assert(isinstance(vol, CloudField))
    vol.commit_info()
    # create test field
    sz = 16
    bbox = BoundingBox(xs=0, xe=16, ys=0, ye=16, mip=0, max_mip=4)
    data = np.ones((1, 2, sz, sz)) 
    f = Field(data, bbox=bbox)
    bcube = BoundingCube.from_bbox(bbox, zs=0, ze=1)
    f.field.y = -f.field.y 
    vol[bcube] = f 
    g = vol[bcube]
    assert(f == g)
    delete_layer('/tmp/cloudvolume/empty_volume')

def test_miplesscloudfield():
    sz = [16, 16, 1]
    info = CloudField.create_new_info(
             num_channels=2, 
             layer_type='image', 
             data_type='int16', 
             encoding='raw',
             resolution=[ 1,1,1 ], 
             voxel_offset=[0,0,0], 
             volume_size=sz,
             chunk_size=sz,
             )
    path = 'file:///tmp/cloudvolume/empty_volume'
    mcv = MiplessCloudField(path, device='cpu', info=info)
    mcv.mkdir()
    assert(os.path.exists('/tmp/cloudvolume/empty_volume'))
    assert(mcv.cloudtype == CloudField)
    assert(isinstance(mcv[0], CloudField))
    mcv[0].cv.add_scale([2,2,1])
    mcv.mkdir()
    mcv = MiplessCloudField(path, device='cpu')
    assert(mcv.info() == info)
    assert(isinstance(mcv[1], CloudField))
    delete_layer('/tmp/cloudvolume/empty_volume')
