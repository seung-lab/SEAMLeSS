import atexit
import os
import signal
import sys
from multiprocessing import Event, Process, Semaphore
from time import sleep, time

from taskqueue import TaskQueue

from args import get_aligner, get_argparser, parse_args
processes = {}


def run_aligner(args, stop_fn=None):
  ppid = os.getppid() # Save parent process ID

  def stop_fn_with_parent_health_check():
    if callable(stop_fn) and stop_fn():
      print("Received stop signal. {} shutting down...".format(os.getpid()))
      return True
    if os.getppid() != ppid:
      print("Parent process is gone. {} shutting down...".format(os.getpid()))
      return True
    return False

  aligner = get_aligner(args)
  with TaskQueue(queue_name=aligner.queue_name, queue_server='sqs', n_threads=0) as tq:
    tq.poll(execute_args=[aligner], stop_fn=stop_fn_with_parent_health_check, 
            lease_seconds=args.lease_seconds)

def create_process(process_id, args):
  stop = Event()
  p = Process(target=run_aligner, args=(args, stop.is_set))
  processes[process_id] = (p, stop)

  # Child process inherits signal handlers from parent, but we want the parent
  # to initiate the cleanup, thus we temporarily replace the signal handlers.
  # NOTE: SIGKILL cannot be ignored, which is fine
  signal.signal(signal.SIGINT, signal.SIG_IGN)
  signal.signal(signal.SIGTERM, signal.SIG_IGN)
  p.start()
  signal.signal(signal.SIGINT, cleanup_processes)
  signal.signal(signal.SIGTERM, cleanup_processes)


def delete_processes(process_ids, timeout=30):
  if not isinstance(process_ids, list):
    process_ids = [process_ids]

  # Send stop event to child processes
  for process_id in process_ids:
    (p, stop) = processes[process_id]
    stop.set()

  # Give them ``timeout`` seconds to finish.
  wait_t = 0.0
  for process_id in process_ids:
    (p, stop) = processes[process_id]
    start_t = time()
    p.join(max(1, timeout - wait_t))
    wait_t += time() - start_t

  # Now kill everything that's still not done and clean up
  for process_id in process_ids:
    (p, stop) = processes[process_id]
    if p.is_alive():
      print("Waited {} s for worker {} to finish - killing it...".format(timeout, process_id))
      os.kill(p.pid, signal.SIGKILL)
    del processes[process_id]


@atexit.register
def cleanup_processes(*args):
  print("Parent process received shutdown signal.")
  delete_processes(list(processes.keys()))
  sys.exit(0)


if __name__ == '__main__':
  parser = get_argparser()
  aligner_args = parse_args(parser)
  process_count = aligner_args.processes
  gpu_process_count = aligner_args.gpu_processes or process_count

  if process_count == 1:
    run_aligner(aligner_args)
  else:
    signal.signal(signal.SIGINT, cleanup_processes)
    signal.signal(signal.SIGTERM, cleanup_processes)

    print("Preparing {} processes...".format(process_count))
    print("GPU will be shared by up to {} processes".format(gpu_process_count))

    aligner_args.gpu_lock = Semaphore(gpu_process_count)
    for process_id in range(process_count):
      create_process(process_id, aligner_args)

    # Check occasionally if our workers are still alive, revive if necessary
    while processes:
      sleep(5.0)
      for process_id in list(processes.keys()):
        (p, stop) = processes[process_id]
        if p.exitcode is None:
          if not p.is_alive():
            # Haven't encountered that one, yet.
            print("Process {} failed to start. Retrying...".format(process_id))
            create_process(process_id, aligner_args)
        elif p.exitcode != 0:
          # Worker got killed unexpectedly - probably an uncaught exception.
          # TODO: If the worker got killed by force while using a gpu_lock
          # we could run into a deadlock.
          print("Process {} terminated with code {}. Restarting...".format(process_id, p.exitcode))
          create_process(process_id, aligner_args)
        else:
          # Never gonna happen, worker will always wait for SQS messages
          print("Process {} finished successfully with code {}.".format(process_id, p.exitcode))
          delete_processes([process_id])
