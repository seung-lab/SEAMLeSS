import pytest

import numpy as np
import torch

from boundingbox import BoundingBox, BoundingCube
from cloudvolume.lib import Vec
from mipless_cloudvolume import MiplessCloudVolume

from fields import Field, FieldCloudVolume
from cloudsample import cloudsample_compose

import shutil
import os

def delete_layer(path):
    if os.path.exists(path):
        shutil.rmtree(path)  

def test_cloudsample():
    sz = Vec(*[4, 4, 1])
    info = FieldCloudVolume.create_new_info(
             num_channels=2, 
             layer_type='image', 
             data_type='int16', 
             encoding='raw',
             resolution=[ 1,1,1 ], 
             voxel_offset=[0,0,0], 
             volume_size=2*sz,
             chunk_size=sz,
             )
    f_path = 'file:///tmp/cloudvolume/empty_volume_f'
    f_vol = FieldCloudVolume(f_path, 
                            as_int16=True, 
                            device='cpu', 
                            mip=0, 
                            info=info, 
                            fill_missing=True)
    f_vol.commit_info()
    g_path = 'file:///tmp/cloudvolume/empty_volume_g'
    g_vol = FieldCloudVolume(g_path, 
                            as_int16=True, 
                            device='cpu', 
                            mip=0, 
                            info=info, 
                            fill_missing=True)
    g_vol.commit_info()
    # create test fields
    def get_field(sz, x_block, y_block):
        bbox = BoundingBox(xs=x_block*sz, xe=(x_block+1)*sz, 
                           ys=y_block*sz, ye=(y_block+1)*sz, 
                           mip=0, max_mip=4)
        data = np.ones((1, 2, sz, sz)) 
        return Field(data, bbox=bbox)
    f = get_field(sz=4, x_block=0, y_block=0)
    f_cube = BoundingCube.from_bbox(f.bbox, zs=0, ze=1)
    f.field.y = 4 
    f.field.x = 0
    f_vol[f_cube] = f 
    g = get_field(sz=4, x_block=0, y_block=1)
    g_cube = BoundingCube.from_bbox(g.bbox, zs=0, ze=1)
    g.field.x = 4
    g.field.y = -4
    g_vol[g_cube] = g 

    mfcv = MiplessCloudVolume(f_path, 
                            mkdir=False, 
                            obj=FieldCloudVolume,
                            fill_missing=True,
                            as_int16=True,
                            device='cpu')
    mgcv = MiplessCloudVolume(g_path, 
                            mkdir=False, 
                            obj=FieldCloudVolume,
                            fill_missing=True,
                            as_int16=True,
                            device='cpu')
    h = cloudsample_compose(f_cv=mfcv,
                            g_cv=mgcv,
                            f_z=0,
                            g_z=0,
                            bbox=f.bbox,
                            f_mip=0,
                            g_mip=0,
                            dst_mip=0,
                            factor=1.,
                            affine=None,
                            pad=0)
    o = get_field(sz=4, x_block=0, y_block=0)
    o_cube = BoundingCube.from_bbox(o.bbox, zs=0, ze=1)
    o.field.x =4 
    o.field.y = 0
    assert(h == o)
    delete_layer('/tmp/cloudvolume/empty_volume_f')
    delete_layer('/tmp/cloudvolume/empty_volume_g')




