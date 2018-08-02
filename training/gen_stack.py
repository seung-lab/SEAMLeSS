from cv_sampler import Sampler
import numpy as np
import h5py
import argparse
import sys
import ast
import time

name = sys.argv[1]
parser = argparse.ArgumentParser()
parser.add_argument('name')
parser.add_argument('--count', type=int)
parser.add_argument('--test', action='store_true')
parser.add_argument('--mip', type=int, default=5)
parser.add_argument('--stack_height', type=int, default=10)
parser.add_argument('--dim', type=int, default=1152)
parser.add_argument('--coords', type=str, default=None)
parser.add_argument('--check_mask', action='store_true')
parser.add_argument('--mask', type=str, default=None)
parser.add_argument('--source', type=str, default='neuroglancer/basil_v0/raw_image_cropped')
parser.add_argument('--zs', type=int, default=1)
parser.add_argument('--ze', type=int, default=1000)
parser.add_argument('--xs', type=int, default=None)
parser.add_argument('--xe', type=int, default=None)
parser.add_argument('--ys', type=int, default=None)
parser.add_argument('--ye', type=int, default=None)
parser.add_argument('--no_split', action='store_true')
args = parser.parse_args()
print(args)

offsets = (args.xs, args.ys) if args.xs is not None and args.ys is not None else None
size = (args.xe, args.ye) if args.xe is not None and args.ye is not None else None

# neuroglancer/basil_v0/father_of_alignment/v3
# neuroglancer/pinky40_v11/image
# neuroglancer/pinky40_alignment/prealigned
# neuroglancer/basil_v0/raw_image

sampler = Sampler(source=('gs://' + args.source), dim=args.dim, mip=args.mip, height=args.stack_height,
                  zs=args.zs, ze=args.ze)
if args.check_mask:
    mask_sampler = Sampler(source=('gs://' + args.mask), dim=args.dim//(2**(5-args.mip)), mip=5,
                           height=args.stack_height, zs=args.zs, ze=args.ze)

def get_chunk(coords=None, coords_=None):
    chunk = None
    if coords is None:
        if not args.check_mask:
            while chunk is None:
                chunk, coords = sampler.random_sample(train=not args.test, offsets=offsets, size=size, split=not args.no_split)
                if chunk is None:
                    print('None')
                    continue
        else:
            while chunk is None:
                mask = None
                while mask is None:
                    mask, coords = mask_sampler.random_sample(train=not args.test, offsets=offsets, size=size, split=not args.no_split)
                    if mask is None:
                        print('None mask')
                        continue
                    maskiness = np.mean(mask < 100)
                    print(maskiness)
                    if maskiness < 0.01:
                        print('Empty mask')
                        mask = None
                        continue
                chunk = sampler.chunk_at_global_coords(coords, (coords[0] + coords[-2], coords[1] + coords[-2], coords[2] + coords[-1]))
                if chunk is None:
                    print('None chunk')
                    continue
                if np.mean(chunk) < 1:
                    print('Zero chunk')
                    chunk = None
                    continue
            return chunk, coords, mask
    else:
        chunk = sampler.chunk_at_global_coords(coords, coords_)
    return chunk, coords

archived_coords = None
if args.coords is not None:
    with open(args.coords) as coord_file:
        archived_coords = ast.literal_eval(coord_file.read())

if archived_coords is not None:
    print('Overwriting argument count ({}) with length of archived coordinates ({})'.format(args.count, len(archived_coords)))
    N = len(archived_coords)
    print('Using fixed coordinates')
else:
    N = args.count

coord_record = []
dataset = np.empty((N, args.stack_height, args.dim, args.dim))
if args.check_mask:
    mask_dataset = np.empty((N, args.stack_height, args.dim // (2**(5-args.mip)), args.dim // (2**(5 - args.mip))))

for i in range(N):
    coords, coords_ = None, None
    if archived_coords is not None:
        ac = archived_coords[i]
        coords = (ac[0], ac[1], ac[2])
        coords_ = (ac[0] + ac[3], ac[1] + ac[3], ac[2] + ac[4])
    if not args.check_mask:
        chunk, coords = get_chunk(coords, coords_)
    else:
        chunk, coords, mask = get_chunk(coords, coords_)
    coord_record.append(coords)
    if chunk is not None:
        dataset[i,:,:,:] = np.transpose(chunk, (2,0,1))
        if args.check_mask:
            mask_dataset[i,:,:,:] = np.transpose(mask, (2,0,1))
    else:
        print('None chunk')
        dataset[i,:,:,:] = 0

    print(i)

record_file = open(args.name + '_' + ('test' if args.test else 'train') + '_mip' + str(args.mip) + 'coords.txt', 'w')
record_file.write(str(coord_record))
record_file.close()

h5f = h5py.File(args.name + '_' + ('test' if args.test else 'train') + '_mip' + str(args.mip) + '.h5', 'w')
h5f.create_dataset('main', data=dataset)

if args.check_mask:
    mask_name = args.mask[(args.mask).rfind('/')+1:]
    print('Adding mask dataset:', mask_name)
    h5f.create_dataset(mask_name, data=mask_dataset)

h5f.close()
