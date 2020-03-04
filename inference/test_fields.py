import pytest

from boundingbox import BoundingBox, BoundingCube
import numpy as np
import torch
from fields import Field, FieldCloudVolume

import shutil
import os

def test_field():
    sz = 16
    bbox = BoundingBox(xs=0, xe=16, ys=0, ye=16, mip=0, max_mip=4)
    data = np.ones((1, 2, sz, sz)) 
    f = Field(data, bbox=bbox)
    assert(f.size == f.bbox.size)
    assert(not f.rel)
    assert(f.mip == 0)
    f = f.down(mips=2)
    assert(f.size == f.bbox.size)
    assert(f.field.shape == (1, 2, 4, 4))
    assert(f.mip == 2)
    f = f.up(mips=1)
    assert(f.mip == 1)
    g = f(f)
    assert(g.size == f.size)
    assert(not g.rel)
    assert(not f.rel)
    assert(g.equal_field(f*2))
    g = f(-f)
    assert(g.size == f.size)
    assert(g.is_identity())
    assert(torch.equal(g.profile(), torch.tensor([[0.,0.]])))

def delete_layer(path):
    if os.path.exists(path):
        shutil.rmtree(path)  

def test_fieldcloudvolume():
    sz = [16, 16, 1]
    info = FieldCloudVolume.create_new_info(
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
    vol = FieldCloudVolume(path, as_int16=True, device='cpu', mip=0, info=info)
    assert(isinstance(vol, FieldCloudVolume))
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




