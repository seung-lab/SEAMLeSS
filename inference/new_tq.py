import gevent.monkey 
gevent.monkey.patch_all(thread=False)

from taskqueue import RegisteredTask, TaskQueue, LocalTaskQueue, GreenTaskQueue

def green_upload(ptask, aligner):
    if aligner.distributed:
        tq = GreenTaskQueue(aligner.queue_name)
        tq.insert_all(ptask, parallel=aligner.threads)
    else:
        tq = LocalTaskQueue(parallel=1)
        tq.insert_all(ptask, args= [aligner])

