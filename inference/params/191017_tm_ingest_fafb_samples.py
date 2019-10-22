from cloudvolume import CloudVolume
from cloudvolume.lib import Bbox, Vec
import tinybrain as tb
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
from time import strftime


src_path = 'gs://fafb_v15_montages/FAFB_montage/v15_montage_20190901_rigid_split/1024x1024'
dst_path = 'gs://seunglab2/FAFBv15/image/samples/003'

src_cv = CloudVolume(src_path, mip=0, fill_missing=True)
for k in range(1,3):
    src_cv.add_scale((2**k,2**k,1))
print(src_cv.info)
dst_size = Vec(2*4096, 2*4096, 30)
dst_start = Vec(46000, 68000, 5020)
src_bbox = Bbox(dst_start, dst_start+dst_size)
dst_bbox = src_cv.bbox_to_mip(src_bbox, 0, 2)
info = CloudVolume.create_new_info(
    num_channels = 1,
    layer_type = 'image',
    data_type = 'uint8',
    encoding = 'raw',
    resolution = [4,4,40],
    voxel_offset = src_bbox.minpt,
    chunk_size = [1024,1024,1],
    volume_size = src_bbox.size3()
)
dst_cv = CloudVolume(dst_path, info=info)
dst_cv.add_scale((2,2,1), chunk_size=[1024,1024,1])
dst_cv.add_scale((4,4,1), chunk_size=[1024,1024,1])
dst_cv.provenance.description = 'Sample of FAFBv15 montages'
dst_cv.provenance.owners = ['tmacrina@princeton.edu']
dst_cv.provenance.processing.append({
    'method': {
        'task': 'ingest',
        'src_path': src_path,
        'dst_path': dst_path,
        'mip': 2,
        'shape': dst_size.tolist(),
        'bounds': [
            dst_bbox.minpt.tolist(),
            dst_bbox.maxpt.tolist(),
            ],
        },
    'by': 'tmacrina@princeton.edu',
    'date': strftime('%Y-%m-%d%H:%M %Z'),
    })
dst_cv.commit_info()
dst_cv.commit_provenance()
dst_cv.mip = 2

def process(z):
    print('Copying & downsample {}'.format(z))
    sbbox = deepcopy(src_bbox)
    sbbox.minpt[2] = z*100
    sbbox.maxpt[2] = z*100+1
    src_img = src_cv[sbbox.to_slices()]
    dst_img = tb.downsample_with_averaging(src_img, (4,4,1))[0]
    dbbox = deepcopy(dst_bbox)
    dbbox.minpt[2] = z
    dbbox.maxpt[2] = z+1
    dst_cv[dbbox.to_slices()] = dst_img

with ProcessPoolExecutor(max_workers=8) as e:
    e.map(process, range(src_bbox.minpt[2], src_bbox.maxpt[2]))

