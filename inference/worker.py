import sys
import torch
from args import get_argparser, parse_args, get_aligner, get_bbox 
#import task_handler # required to initialize RegisteredTasks

from taskqueue import TaskQueue

if __name__ == '__main__':
  parser = get_argparser()
  args = parse_args(parser) 
  aligner = get_aligner(args)

  with TaskQueue(queue_name=args.queue_name, queue_server='sqs', n_threads=0) as tq:
    tq.poll(execute_args=[ aligner ])
    #tq.poll(execute_args=[ aligner ], verbose=True)


