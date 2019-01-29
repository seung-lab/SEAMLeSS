import sys
import torch
from args import get_argparser, parse_args, get_aligner, get_bbox
#import task_handler # required to initialize RegisteredTasks
import os
from time import sleep
from multiprocessing import Process, Event
import signal
import atexit

from taskqueue import TaskQueue

processes = {}


def run_aligner(aligner, stop_fn=None):
  with TaskQueue(queue_name=aligner.queue_name, queue_server='sqs', n_threads=0) as tq:
    tq.poll(execute_args=[aligner], stop_fn=stop_fn)


def create_process(process_id, aligner, delay_start=False):
  stop = Event()
  p = Process(target=run_aligner, args=(aligner, stop.is_set))
  processes[process_id] = (p, aligner, stop)
  if not delay_start:
    start_process(process_id)


def start_process(process_id):
  # Child process inherits signal handlers from parent, but we want the parent
  # to do the clean up, thus we temporarily replace the signal handlers.
  # NOTE: SIGKILL cannot be ignored, which is fine
  signal.signal(signal.SIGINT, signal.SIG_IGN)
  signal.signal(signal.SIGTERM, signal.SIG_IGN)
  (p, _, _) = processes[process_id]
  p.start()
  signal.signal(signal.SIGINT, cleanup_processes)
  signal.signal(signal.SIGTERM, cleanup_processes)


def delete_process(process_id, timeout=30):
  # Tell worker to stop within the next ``timeout`` seconds, then kill it.
  (p, _, stop) = processes[process_id]
  print("Attempting to stop worker {}...".format(process_id))
  stop.set()
  p.join(timeout)
  if p.is_alive():
    print("Waited {} s for worker {} to finish - killing it...".format(timeout, process_id))
    os.kill(p.pid, signal.SIGKILL)
  del processes[process_id]


@atexit.register
def cleanup_processes(*args):
  print("Parent process received shutdown signal.")
  for process_id in list(processes.keys()):
    delete_process(process_id)
  sys.exit(0)


if __name__ == '__main__':
  parser = get_argparser()
  args = parse_args(parser)

  if args.processes == 1:
    aligner = get_aligner(args)
    run_aligner(aligner)
  else:
    signal.signal(signal.SIGINT, cleanup_processes)
    signal.signal(signal.SIGTERM, cleanup_processes)

    print("Preparing {} processes...".format(args.processes))
    for process_id in range(args.processes):
      aligner = get_aligner(args)
      create_process(process_id, aligner, delay_start=True)

    for process_id in processes:
      start_process(process_id)

    # Check occasionally if our workers are still alive, revive if necessary
    while processes:
      sleep(5.0)
      for process_id in list(processes.keys()):
        (p, aligner, stop) = processes[process_id]
        if p.exitcode is None:
          if not p.is_alive():
            # Haven't encountered that one, yet.
            print("Process {} failed to start. Retrying...".format(process_id))
            create_process(process_id, aligner)
        elif p.exitcode != 0:
          # Worker got killed unexpectedly - probably an uncaught exception.
          print("Process {} terminated with code {}. Restarting...".format(process_id, p.exitcode))
          create_process(process_id, aligner)
        else:
          # Never gonna happen, worker will always wait for SQS messages
          print("Process {} finished successfully with code {}.".format(process_id, p.exitcode))
          delete_process(process_id)
