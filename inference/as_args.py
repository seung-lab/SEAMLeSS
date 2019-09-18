from mp_aligner import Aligner, BoundingBox
from getpass import getuser
import argparse

def positive_int(val):
  ival = int(val)
  if ival <= 0:
    raise argparse.ArgumentTypeError("Positive integer expected, got %s" % val)
  return ival

def get_argparser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--queue_name', type=str, default=None)
  parser.add_argument('--processes', type=positive_int, default=1,
     help='no. of processes to spawn on a single worker')
  parser.add_argument('--gpu_processes', type=positive_int, default=None,
     help='max no. of processes that might share the GPU at any given time')
  parser.add_argument('--threads', type=int, default=1,
     help='no. of threads to use in scheduling chunks (locally & distributed)')
  parser.add_argument('--task_batch_size', type=int, default=1,
     help='no. of tasks to group together for a single worker')
  parser.add_argument('--lease_seconds', type=int, default=43200,
     help='no. of seconds that polling will lease a task before it becomes visible again')
  parser.add_argument('--dry_run', 
     help='prevent task executes, but allow task print outs',
     action='store_true')
  parser.add_argument('--IO_timeout', type=positive_int, default=None,
                      help='timeout for I/O operations')

  return parser

def parse_args(parser, arg_string=''):
  if arg_string:
    args = parser.parse_args(arg_string)
  else:
    args = parser.parse_args()
  return args

def get_aligner(args):
  """Create Aligner object from args
  """
  return Aligner(**vars(args))

def get_bbox(args):
  """Create BoundingBox object from args
  """
  # interleave coords by flattening
  coords = [x for t in zip(args.bbox_start[:2], args.bbox_stop[:2]) for x in t]
  return BoundingBox(*coords, mip=args.bbox_mip, max_mip=args.max_mip)

def get_provenance(args):
  args.user = getuser()
  args.project = 'seamless'
  return vars(args)
