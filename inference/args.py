from aligner import Aligner, BoundingBox
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
     help='no. of processes to use on a single worker (useful to bypass GKE GPU limit)')
  parser.add_argument('--threads', type=int, default=1,
     help='no. of threads to use in scheduling chunks (locally & distributed)')
  parser.add_argument('--task_batch_size', type=int, default=1,
     help='no. of tasks to group together for a single worker')
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
  return BoundingBox(*coords, mip=0, max_mip=args.max_mip)

def get_provenance(args):
  args.user = getuser()
  args.project = 'seamless'
  return vars(args)
