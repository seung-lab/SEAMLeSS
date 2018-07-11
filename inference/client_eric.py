import sys
from aligner import Aligner, BoundingBox
from link_builder import ng_link
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str)
parser.add_argument('--model_path', type=str)
parser.add_argument('--out_name', type=str)
parser.add_argument('--mip', type=int)
parser.add_argument('--render_mip', type=int)
parser.add_argument('--should_contrast', type=int)
parser.add_argument('--num_targets', type=int)
parser.add_argument('--edge_crop', type=int, default=384)
parser.add_argument('--max_displacement', type=int, default=2048)
parser.add_argument('--max_mip', type=int, default=9)
parser.add_argument('--xs', type=int)
parser.add_argument('--xe', type=int)
parser.add_argument('--ys', type=int)
parser.add_argument('--ye', type=int)
parser.add_argument('--stack_size', type=int, default=100)
parser.add_argument('--zs', type=int)
args = parser.parse_args()

out_name = args.out_name
mip = args.mip
render_mip = args.render_mip
should_contrast = bool(args.should_contrast)
num_targets = args.num_targets
model_path = args.model_path
model_name = model_path[model_path.rindex('/')+1:model_path.rindex('.')]
max_displacement = args.max_displacement
edge_crop  = args.edge_crop
mip_range = (mip,mip)
high_mip_chunk = (1024, 1024)
source = args.source
max_mip = args.max_mip
xs = args.xs
xe = args.xe
ys = args.ys
ye = args.ye
zs = args.zs
v_off = (xs, ys, zs)
x_size = xe - xs
y_size = ye - ys
stack_size = args.stack_size

if num_targets < 1:
    print('num_targets must be > 0')
    sys.exit(1)

print('Model name:', model_name)
print('Tag:', out_name)
print('Source:', source)
print('Coordinates:', (args.xs, args.ys, args.zs), (args.xe, args.ye, args.zs+args.stack_size))
print('Mip:', mip)
print('Contrast:', should_contrast)
print('Max mip:', max_mip-1)
print('NG link:', ng_link(out_name, 'precomputed://' + 'gs://neuroglancer/seamless/' + model_name+'_'+out_name+'/image', source[source.rindex('/')+1:], 'precomputed://' + source, (xs+xe)//2, (ys+ye)//2, zs))

a = Aligner(model_path, max_displacement, edge_crop, mip_range, high_mip_chunk, source,
            'gs://neuroglancer/seamless/{}_{}'.format(model_name, out_name), render_low_mip=render_mip,
            skip=0, topskip=0, should_contrast=should_contrast, num_targets=num_targets)

bbox = BoundingBox(v_off[0], v_off[0]+x_size, v_off[1], v_off[1]+y_size, mip=0, max_mip=max_mip)
print(bbox.x_range(mip=0), bbox.y_range(mip=0))

stack_start = v_off[2]
a.align_ng_stack(stack_start, stack_start+stack_size, bbox, move_anchor=True)

