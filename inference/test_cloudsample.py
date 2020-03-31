import pytest

import numpy as np
import torch

from boundingbox import BoundingBox, BoundingCube
from cloudvolume.lib import Vec


from fields import Field
from cloudtensor import CloudField, MiplessCloudField
from cloudsample import cloudsample_compose

import shutil
import os

def delete_layer(path):
    if os.path.exists(path):
        shutil.rmtree(path)  

def test_cloudsample():
    sz = Vec(*[4, 4, 1])
    info = CloudField.create_new_info(
             num_channels=2, 
             layer_type='image', 
             data_type='int16', 
             encoding='raw',
             resolution=[ 1,1,1 ], 
             voxel_offset=[-4,-4,0], 
             volume_size=4*sz,
             chunk_size=sz,
             )
    f_path = 'file:///tmp/cloudvolume/empty_volume_f'
    f_vol = CloudField(f_path, 
                            device='cpu', 
                            mip=0, 
                            info=info, 
                            fill_missing=True)
    f_vol.commit_info()
    g_path = 'file:///tmp/cloudvolume/empty_volume_g'
    g_vol = CloudField(g_path, 
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
    f.field.x = 0
    f.field.y = 4 
    f_vol[f_cube] = f 
    g = get_field(sz=4, x_block=0, y_block=1)
    g_cube = BoundingCube.from_bbox(g.bbox, zs=0, ze=1)
    g.field.x = 4
    g.field.y = -4
    g_vol[g_cube] = g 

    mfcv = MiplessCloudField(f_path, 
                            device='cpu',
                            fill_missing=True)
    mgcv = MiplessCloudField(g_path, 
                            device='cpu',
                            fill_missing=True)
    h = cloudsample_compose(f_cv=mfcv,
                            g_cv=mgcv,
                            f_z=0,
                            g_z=0,
                            bbox=f.bbox,
                            f_mip=0,
                            g_mip=0,
                            dst_mip=0,
                            factor=1.,
                            pad=0)
    o = get_field(sz=4, x_block=0, y_block=0)
    o_cube = BoundingCube.from_bbox(o.bbox, zs=0, ze=1)
    o.field.x = 4 
    o.field.y = 0
    assert(h.allclose(o, atol=1e-6))

    f1 = Field(np.ones((1,2,16,16)), BoundingBox(xs=-4, xe=12, 
                                                 ys=-4, ye=12,
                                                 mip=0, max_mip=4))
    f1_cube = BoundingCube.from_bbox(f1.bbox, zs=0, ze=1)
    f1.field.x = 0
    f1.field.y = 4 
    f_vol[f1_cube] = f1 
    g1 = Field(np.ones((1,2,16,16)), BoundingBox(xs=-4, xe=12, 
                                                 ys=-4, ye=12,
                                                 mip=0, max_mip=4))
    g1_cube = BoundingCube.from_bbox(g1.bbox, zs=0, ze=1)
    g1.field.x = 4
    g1.field.y = -4 
    g_vol[g1_cube] = g1 
    h = cloudsample_compose(f_cv=mfcv,
                            g_cv=mgcv,
                            f_z=0,
                            g_z=0,
                            bbox=BoundingBox(xs=0, xe=4,
                                             ys=0, ye=4,
                                             mip=0, max_mip=4),
                            f_mip=0,
                            g_mip=0,
                            dst_mip=0,
                            factor=1.,
                            pad=1)
    o = get_field(sz=4, x_block=0, y_block=0)
    o_cube = BoundingCube.from_bbox(o.bbox, zs=0, ze=1)
    o.field.x = 4 
    o.field.y = 0
    assert(h.allclose(o, atol=1e-6))
    delete_layer('/tmp/cloudvolume/empty_volume_f')
    delete_layer('/tmp/cloudvolume/empty_volume_g')

