import argparse
from cloudvolume import CloudVolume
from cloudvolume.lib import Vec, Bbox
import numpy as np
import h5py

def cutout_to_h5(vol, bbox, h5, name, dtype=np.uint8):
    """Make CloudVolume cutout and store in H5

    Args:
        vol: CloudVolume object
        bbox: CloudVolume Bbox of cutout
        h5: h5py File object
        name: str for H5 dataset name
        dtype: Datatype of cutout in H5

    Returns:
        None. Stores cutout in (batch)xZxHxW shape
    """
    data = vol[bbox.to_slices()]
    data = np.transpose(data, (3,2,1,0))
    print(data.shape)
    h5.create_dataset(name, data=data.astype(dtype))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path',
        type=str,
        help='CloudVolume path where image will be sourced')
    parser.add_argument('--dst_path',
        type=str,
        help='Local path where output H5 will be stored')
    parser.add_argument('--mip',
        type=int,
        default=0,
        help='MIP level for image source')
    parser.add_argument('--bbox_start',
        nargs=3,
        type=int,
        help='bbox origin, 3-element int list')
    parser.add_argument('--bbox_stop',
        nargs=3,
        type=int,
        help='bbox origin+shape, 3-element int list')
    parser.add_argument('--bbox_mip',
        type=int,
        default=0,
        help='MIP level at which bbox_start & bbox_stop are specified')
    parser.add_argument('--dtype',
        type=str,
        default='np.uint8',
        help='Datatype (converted via eval)')
    args = parser.parse_args()
    bbox = Bbox(args.bbox_start, args.bbox_stop)
    src = CloudVolume(args.src_path, mip=args.mip)
    src_bbox = src.bbox_to_mip(bbox, args.bbox_mip, args.mip)
    h5 = h5py.File(args.dst_path, 'w')
    cutout_to_h5(src, src_bbox, h5, 'main', eval(args.dtype))
    h5.close()

