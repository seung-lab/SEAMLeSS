import sys
import torch
from args import get_argparser, parse_args, get_aligner, get_bbox 
from os.path import join

if __name__ == '__main__':
  parser = get_argparser()
  args = parse_args(parser) 
  a = get_aligner(args)

  a.listen_for_tasks()
