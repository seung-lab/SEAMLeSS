import pytest

import numpy as np
import torch

from boundingbox import BoundingBox, BoundingCube
from cloudvolume.lib import Vec

from fields import Field, FieldCloudVolume
from pairwise_fields import PairwiseFields, PairwiseVoteTask 

import shutil
import os

def delete_layer(path):
    if os.path.exists(path):
        shutil.rmtree(path)  

def test_pairwise_fields():
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


def test_pairwise_vote_task():
    mip=0
    sz = Vec(*[4, 4, 1])
    info = FieldCloudVolume.create_new_info(
             num_channels=2, 
             layer_type='image', 
             data_type='float32', 
             encoding='raw',
             resolution=[ 1,1,1 ], 
             voxel_offset=[0,0,0], 
             volume_size=[8,8,4],
             chunk_size=sz,
             )

    def get_field(sz, x_block, y_block):
        bbox = BoundingBox(xs=x_block*sz, xe=(x_block+1)*sz, 
                           ys=y_block*sz, ye=(y_block+1)*sz, 
                           mip=mip, max_mip=4)
        data = np.ones((1, 2, sz, sz)) 
        return Field(data, bbox=bbox)

    f = get_field(sz=4, x_block=0, y_block=0)
    offsets = [-3,-2,-1,1,2,3]
    estimates_path = 'file:///tmp/cloudvolume/estimates'
    F = PairwiseFields(path=estimates_path,
                       offsets=offsets,
                       bbox=f.bbox,
                       mip=mip,
                       pad=0,
                       mkdir=True,
                       as_int16=False,
                       device='cpu',
                       info=info,
                       fill_missing=False)
    corrected_path = 'file:///tmp/cloudvolume/corrected'
    C = PairwiseFields(path=corrected_path,
                       offsets=offsets,
                       bbox=f.bbox,
                       mip=mip,
                       pad=0,
                       mkdir=True,
                       as_int16=False,
                       device='cpu',
                       info=info,
                       fill_missing=False)

    # Want vector voting to be on three vectors that are each rotated by 2*\pi / 3
    # Translate so that average is (0, 1)

    # F[(1, 0)] = (0,4)
    f = get_field(sz=4, x_block=0, y_block=0)
    f_cube = BoundingCube.from_bbox(f.bbox, zs=0)
    f.field.x = 0 
    f.field.y = 4
    # convention: PairwiseField.cvs[OFFSET][MIP][BoundingCube]
    F.cvs[1][mip][f_cube] = f 
    # F[(1, 2)] = (0,4)
    f_cube = BoundingCube.from_bbox(f.bbox, zs=2)
    F.cvs[-1][mip][f_cube] = f 
    # F[(1, 3)] = (0,4)
    f_cube = BoundingCube.from_bbox(f.bbox, zs=3)
    F.cvs[-2][mip][f_cube] = f 
    # F[(2, 0), new_bbox] = (2*sqrt(3), -2-3)
    # F[(1, 2, 0)] = (2*sqrt(3), -2)
    f = get_field(sz=4, x_block=0, y_block=1)
    f_cube = BoundingCube.from_bbox(f.bbox, zs=0)
    f.field.x = 2*np.sqrt(3)
    f.field.y = -5
    F.cvs[2][mip][f_cube] = f 
    # F[(3, 0), new_bbox] = (-2*sqrt(3), -2-3)
    # F[(1, 3, 0)] = (-2*sqrt(3), -2)
    f = get_field(sz=4, x_block=0, y_block=1)
    f_cube = BoundingCube.from_bbox(f.bbox, zs=0)
    f.field.x = -2*np.sqrt(3)
    f.field.y = -5
    F.cvs[3][mip][f_cube] = f 

    f = get_field(sz=4, x_block=0, y_block=0)
    VoteTask = PairwiseVoteTask(estimates_path=estimates_path,
                                 corrected_path=corrected_path,
                                 weights_path='',
                                 offsets=offsets,
                                 src_z=0,
                                 tgt_offsets=[1,2,3],
                                 bbox=f.bbox.serialize(),
                                 mip=mip,
                                 pad=0,
                                 as_int16=False,
                                 device='cpu',
                                 softmin_temp=10000, # make all vectors equal
                                 blur_sigma=1)
    # We'll vote for F[(1,0)], but the rest will have errors
    # with pytest.raises(ValueError) as e:
    VoteTask.execute()

    o = get_field(sz=4, x_block=0, y_block=0)
    o.field.x = 0 
    o.field.y = 2/3
    x = C[(1,0)]
    assert(o.allclose(x, atol=1e-3))

    delete_layer('/tmp/cloudvolume/estimates')
    delete_layer('/tmp/cloudvolume/corrected')

