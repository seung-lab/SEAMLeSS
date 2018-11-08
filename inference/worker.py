import sys
from aligner import Aligner, BoundingBox
from link_builder import ng_link
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str)
parser.add_argument('--size', type=int, default=8)
parser.add_argument('--skip', type=int, default=0)
parser.add_argument('--model_path', type=str)
parser.add_argument('--out_name', type=str)
parser.add_argument('--queue_name', type=str, default=None)
parser.add_argument('--mip', type=int)
parser.add_argument('--render_mip', type=int)
parser.add_argument('--should_contrast', type=int)
parser.add_argument('--num_targets', type=int)
parser.add_argument('--edge_pad', type=int, default=384)
parser.add_argument('--max_displacement',
  help='the size of the largest displacement expected; should be 2^high_mip',
  type=int, default=2048)
parser.add_argument('--max_mip', type=int, default=9)
parser.add_argument('--xs', type=int)
parser.add_argument('--xe', type=int)
parser.add_argument('--ys', type=int)
parser.add_argument('--ye', type=int)
parser.add_argument('--stack_size', type=int, default=100)
parser.add_argument('--zs', type=int)
parser.add_argument('--thread', type=int, default=1)
parser.add_argument('--no_anchor', action='store_true')
parser.add_argument('--p_render', help='parallel rendering among all slices', action='store_true')
parser.add_argument('--no_flip_average',
  help='disable flip averaging, on by default (flip averaging is used to eliminate drift)',
  action='store_true')
parser.add_argument('--run_pairs',
  help='only run on consecutive pairs of input slices, rather than sequentially aligning a whole stack',
  action='store_true')
parser.add_argument('--write_intermediaries',
  help='write encodings, residuals, & cumulative residuals to cloudvolumes',
  action='store_true')
parser.add_argument('--upsample_residuals',
  help='upsample residuals & cumulative residuals when writing intermediaries; requires --write_intermediaries flag',
  action='store_true')
parser.add_argument('--old_upsample', help='revert to the old pytorch upsampling (using align_corners=True)', action='store_true')
parser.add_argument('--old_vectors', help='expect the net to return vectors in the old vector field convention, '
  'where -1 and 1 refer to the centers of the border pixels rather than the image edges.',
  action='store_true')

args = parser.parse_args()

out_name = args.out_name
mip = args.mip
render_mip = args.render_mip
should_contrast = bool(args.should_contrast)
num_targets = args.num_targets
model_path = args.model_path
model_name = model_path[model_path.rindex('/')+1:model_path.rindex('.')]
max_displacement = args.max_displacement
edge_pad  = args.edge_pad
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
out_cv = 'gs://neuroglancer/seamless/{}_{}'.format(model_name, out_name)
thread = args.thread
if num_targets < 1:
    print('num_targets must be > 0')
    sys.exit(1)

print('Model name:', model_name)
print('Tag:', out_name)
print('Source:', source)
print('Coordinates:', (args.xs, args.ys, args.zs), (args.xe, args.ye, args.zs+args.stack_size))
print('Mip:', mip)
print('Contrast:', should_contrast)
print('Max mip:', max_mip)
print('NG link:', ng_link(out_name, 'precomputed://' + 'gs://neuroglancer/seamless/' + model_name+'_'+out_name+'/image', source[source.rindex('/')+1:], 'precomputed://' + source, (xs+xe)//2, (ys+ye)//2, zs))

a = Aligner(model_path, max_displacement, edge_pad, mip_range, high_mip_chunk,
            source, out_cv, render_low_mip=render_mip, render_high_mip=max_mip, threads=thread
            skip=args.skip, topskip=0, size=args.size, should_contrast=should_contrast,
            num_targets=num_targets, flip_average=not args.no_flip_average,
            run_pairs=args.run_pairs,
            write_intermediaries=args.write_intermediaries,
            upsample_residuals=args.upsample_residuals, old_upsample=args.old_upsample,
            old_vectors=args.old_vectors, queue_name=args.queue_name, p_render=args.p_render)
bbox = BoundingBox(v_off[0], v_off[0]+x_size, v_off[1], v_off[1]+y_size, mip=0, max_mip=max_mip)

a.listen_for_tasks(v_off[2], stack_size, bbox)
