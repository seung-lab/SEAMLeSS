import pytest

import numpy as np
import torch

from boundingbox import BoundingBox, BoundingCube
from cloudvolume.lib import Vec

from fields import Field, FieldCloudVolume
from pairwise_fields import PairwiseFields 

import shutil
import os

def delete_layer(path):
    if os.path.exists(path):
        shutil.rmtree(path)  

def test_cloudsample():
    mip=0
    sz = Vec(*[4, 4, 1])
    info = FieldCloudVolume.create_new_info(
             num_channels=2, 
             layer_type='image', 
             data_type='int16', 
             encoding='raw',
             resolution=[ 1,1,1 ], 
             voxel_offset=[0,0,-1], 
             volume_size=[8,8,3],
             chunk_size=sz,
             )

    def get_field(sz, x_block, y_block):
        bbox = BoundingBox(xs=x_block*sz, xe=(x_block+1)*sz, 
                           ys=y_block*sz, ye=(y_block+1)*sz, 
                           mip=mip, max_mip=4)
        data = np.ones((1, 2, sz, sz)) 
        return Field(data, bbox=bbox)

    g = get_field(sz=4, x_block=0, y_block=0)
    F = PairwiseFields(path='file:///tmp/cloudvolume/empty_volume',
                       offsets=[-2,-1,1],
                       bbox=g.bbox,
                       mip=mip,
                       pad=0,
                       mkdir=True,
                       as_int16=True,
                       device='cpu',
                       info=info,
                       fill_missing=True)
    assert(F.cvs[-2].path == 'file:///tmp/cloudvolume/empty_volume/-2')
    assert(F.cvs[-1].path == 'file:///tmp/cloudvolume/empty_volume/-1')
    assert(F.cvs[1].path == 'file:///tmp/cloudvolume/empty_volume/1')
    assert(os.path.exists('/tmp/cloudvolume/empty_volume/-1'))
    assert(os.path.exists('/tmp/cloudvolume/empty_volume/1'))

    with pytest.raises(ValueError) as e:
        F[(-1,0,1)] = g

    # F[(1, 0)]
    f = get_field(sz=4, x_block=0, y_block=1)
    f_cube = BoundingCube.from_bbox(f.bbox, zs=0)
    f.field.x = 4 
    f.field.y = 0
    # convention: PairwiseField.cvs[OFFSET][MIP][BoundingCube]
    F.cvs[1][mip][f_cube] = f 
    # can't access this with our current PairwiseField, because it's in a different bbox
    assert(F[(1,0)].is_identity())

    # F[(-1,1)]
    g_cube = BoundingCube.from_bbox(g.bbox, zs=1)
    g.field.x = 0
    g.field.y = 4 
    F.cvs[-2][mip][g_cube] = g 
    assert(F[(-1,1)] == g)

    o = get_field(sz=4, x_block=0, y_block=0)
    o.field.x = 4
    o.field.y = 4 
    x = F[(-1,1,0)]
    assert(x == o)

    with pytest.raises(ValueError) as e:
        x = F[(-1,)]
    with pytest.raises(ValueError) as e:
        x = F[(-3,1)]

    delete_layer('/tmp/cloudvolume/empty_volume')




