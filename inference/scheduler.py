from time import time, sleep
from concurrent.futures import ProcessPoolExecutor
from threading import Lock

import taskqueue
from taskqueue import GreenTaskQueue, LocalTaskQueue
from boundingbox import BoundingBox

import boto3
import tenacity

retry = tenacity.retry(
  reraise=True, 
  stop=tenacity.stop_after_attempt(7), 
  wait=tenacity.wait_full_jitter(0.5, 60.0),
)

class Scheduler():
    """Break an operation into tasks & monitor completion
    
    Progeny of the Aligner class
    """
    def __init__(self, queue_name=None, threads=1, **kwargs):
        self.queue_name = queue_name
        self.sqs = None
        self.queue_url = None
        self.threads = threads
        self.gpu_lock = kwargs.get('gpu_lock', None)  # multiprocessing.Semaphore

    @property
    def distributed(self):
        """Are we scheduling these tasks to a remote queue?
        """
        return self.queue_name is not None

    def get_chunks(self, bbox, chunk_size, voxel_offset, mip, max_mip=12):
        """Break bbox into list of chunks with chunk_size, given offset for all data 

        Args:
           bbox: BoundingBox for region to be broken into chunks
           chunk_size: tuple for dimensions of chunk that bbox will be broken into;
             will be set to min(chunk_size, self.chunk_size)
           voxel_offset: tuple for x,y origin for the entire dataset, from which chunks
             will be aligned
           mip: int for MIP level at which bbox is defined
           max_mip: int for the maximum MIP level at which the bbox is valid
        """
        raw_x_range = bbox.x_range(mip=mip)
        raw_y_range = bbox.y_range(mip=mip)
        
        x_chunk = chunk_size[0]
        y_chunk = chunk_size[1]
        
        x_offset = voxel_offset[0]
        y_offset = voxel_offset[1]
        x_remainder = ((raw_x_range[0] - x_offset) % x_chunk)
        y_remainder = ((raw_y_range[0] - y_offset) % y_chunk)

        calign_x_range = [raw_x_range[0] - x_remainder, raw_x_range[1]]
        calign_y_range = [raw_y_range[0] - y_remainder, raw_y_range[1]]

        chunks = []
        for xs in range(calign_x_range[0], calign_x_range[1], chunk_size[0]):
            for ys in range(calign_y_range[0], calign_y_range[1], chunk_size[1]):
                chunks.append(BoundingBox(xs, xs + chunk_size[0],
                                         ys, ys + chunk_size[1],
                                         mip=mip, max_mip=max_mip))
        return chunks

    @retry
    def sqs_is_empty(self):
        """Is our remote queue empty?
        """
        attribute_names = ['ApproximateNumberOfMessages', 'ApproximateNumberOfMessagesNotVisible']
        responses = []
        for i in range(3):
            response = self.sqs.get_queue_attributes(QueueUrl=self.queue_url,
                                                     AttributeNames=attribute_names)
            for a in attribute_names:
                responses.append(int(response['Attributes'][a]))
                print('{}     '.format(responses[-2:]), end="\r", flush=True)
            if i < 2:
              sleep(1)
        return all(i == 0 for i in responses)

    def wait_for_sqs_empty(self):
        """Stop until our remote queue is empty
        """
        self.sqs = boto3.client('sqs', region_name='us-east-1')
        self.queue_url  = self.sqs.get_queue_url(QueueName=self.queue_name)["QueueUrl"]
        print("\nSQS Wait")
        print("No. of messages / No. not visible")
        sleep(5)
        while not self.sqs_is_empty():
          sleep(1)

    def remote_upload(self, tasks):
        """Fast ingest of tasks to our remote queue
        """
        with GreenTaskQueue(queue_name=self.queue_name) as tq:
            tq.insert_all(tasks)  

    def execute(self, task_iterator, z_range):
        """Break operation into tasks and wait for tasks to finish
        """
        if len(z_range) > 0:
            ptask = []
            range_list = self.make_range(z_range)
            start = time()

        for irange in range_list:
            ptask.append(task_iterator(irange))
        if self.distributed:
            with ProcessPoolExecutor(max_workers=self.threads) as executor:
                executor.map(self.remote_upload, ptask)
        else:
            for t in ptask:
                tq = LocalTaskQueue(parallel=1)
                tq.insert_all(t)
 
        end = time()
        diff = end - start
        print('Sending {} use time: {}'.format(task_iterator, diff))
        if self.distributed:
            print('Run {}'.format(task_iterator))
            # wait
            start = time()
            self.wait_for_sqs_empty()
            end = time()
            diff = end - start
            print('Executing {} use time: {}\n'.format(task_iterator, diff))

    def make_range(self, block_range):
        """Split a range of tasks across threads into multiple ranges
        """
        part_num = self.threads
        rangelen = len(block_range)
        if(rangelen < part_num):
            srange =1
            part = rangelen
        else:
            part = part_num
            srange = rangelen//part
        range_list = []
        for i in range(part-1):
            range_list.append(block_range[i*srange:(i+1)*srange])
        range_list.append(block_range[(part-1)*srange:])
        return range_list

