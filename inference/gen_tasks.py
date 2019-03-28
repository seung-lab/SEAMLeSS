import gevent.monkey
gevent.monkey.patch_all(thread=False)
from concurrent.futures import ProcessPoolExecutor
import taskqueue
from taskqueue import TaskQueue, GreenTaskQueue
import pathos.pools
import copy
import tasks
from boundingbox import BoundingBox, deserialize_bbox

def make_range(block_range, part_num):
  rangelen = len(block_range)
  if(rangelen < part_num):
      srange =1
      part = rangelen
  else:
      part = part_num
      srange = rangelen//part
  if rangelen%2 == 0:
      odd_even = 0
  else:
      odd_even = 1
  range_list = []
  for i in range(part-1):
      range_list.append(block_range[i*srange:(i+1)*srange + 1])
  range_list.append(block_range[(part-1)*srange:])
  return range_list, odd_even

def remote_upload(tasks):
    with GreenTaskQueue(queue_name=args.queue_name) as tq:
       tq.insert_all(tasks) 


def break_into_chunks(bbox, chunk_size, offset, mip, max_mip=12):
  """Break bbox into list of chunks with chunk_size, given offset for all data
  Args:
     bbox: BoundingBox for region to be broken into chunks
     chunk_size: tuple for dimensions of chunk that bbox will be broken into;
       will be set to min(chunk_size, self.chunk_size)
     offset: tuple for x,y origin for the entire dataset, from which chunks
       will be aligned
     mip: int for MIP level at which bbox is defined
     max_mip: int for the maximum MIP level at which the bbox is valid
  """

  raw_x_range = bbox.x_range(mip=mip)
  raw_y_range = bbox.y_range(mip=mip)

  x_chunk = chunk_size[0]
  y_chunk = chunk_size[1]

  x_offset = offset[0]
  y_offset = offset[1]

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

def make_copy_tasks(a, copy_range, block_range, block_types, bbox_lookup, cm, src,
                    dsts, mip, mask_cv, mask_mip, mask_val):
  import pathos_issue
  from taskqueue import PrintTask
  class TaskIterator():
      def __init__(self, brange, odd_even):
          self.brange = brange
          self.odd_even = odd_even
      def __len__(self):
          return len(self.brange)
      def __getitem__(self, slc):
          x = copy.deepcopy(self)
          x.brange =  self.brange
          x.odd_even = self.odd_even
          return x
      def __iter__(self):
          for block_offset in copy_range:
            prefix = block_offset
            #for i, block_start in enumerate(block_range):
            for i, block_start in enumerate(self.brange):
              block_type = block_types[(i + self.odd_even) % 2]
              #block_type = block_types[i % 2]
              dst = dsts[block_type]
              z = block_start + block_offset
              bbox = bbox_lookup[z]
              #t = a.copy(cm, src, dst, z, z, bbox, mip, is_field=False,
              #           mask_cv=mask_cv, mask_mip=mask_mip, mask_val=mask_val,
              #           prefix=prefix)
              #chunks = break_into_chunks(bbox, cm.dst_chunk_sizes[mip],
              #                      cm.dst_voxel_offsets[mip], mip=mip, 
              #                      max_mip=cm.max_mip) 
              #for chunk in chunks:
              #    yield tasks.CopyTask(src, dst, z, z, chunk, mip, is_field=False,
              #                         mask_cv=mask_cv, mask_mip=mask_mip,
              #                         mask_val=mask_val, prefix=prefix)
              #yield from t
              #yield PrintTask(str(i)) 
  
  ptask = TaskIterator(block_range, 0)
  #p_module = ptask.__class__.__module__
  #ptask.__class__.__module__ = '__main__'
  #range_list, odd_even = make_range(block_range, a.threads)
  #for i, irange in enumerate(range_list):
  #    ptask.append(TaskIterator(irange, i*odd_even))
  tq = GreenTaskQueue('deepalign_zhen')
  tq.insert_all(ptask, parallel=a.threads) 
  #tq.insert_all(ptask, nt) 
  
  #ptask.__class__.__module__ = p_module
  #range_list, odd_even = make_range(block_range, a.threads)
  #with ProcessPoolExecutor(max_workers=1) as executor:
  #with pathos.pools.ProcessPool(1) as executor:
  #    executor.map(remote_upload, ptask)

  #new_run(a, ptask)
  #return ptask
  #return TaskIterator()

def make_computeField_tasks(a, z_offset, block_offset, block_range, block_types,
                            cm, model_path, src, dsts, serial_field, bbox, mip,
                            pad, src_mask_cv, src_mask_mip, src_mask_val,
                            tgt_mask_cv, tgt_mask_mip, tgt_mask_val):
  class TaskIterator(object):
      def __init__(self, brange, odd_even):
          self.brange = brange
          self.odd_even = odd_even
      def __iter__(self):
          prefix = block_offset
          for i, block_start in enumerate(self.brange):
              block_type = block_types[(i+self.odd_even) % 2]
              dst = dsts[block_type]
              z = block_start + block_offset 
              t = a.compute_field(cm, args.model_path, src, dst, serial_field, 
                                  z, z+z_offset, bbox, mip, pad, src_mask_cv=src_mask_cv,
                                  src_mask_mip=src_mask_mip, src_mask_val=src_mask_val,
                                  tgt_mask_cv=src_mask_cv, tgt_mask_mip=src_mask_mip, 
                                  tgt_mask_val=src_mask_val, prefix=prefix)
              yield from t
  ptask = []
  range_list, odd_even = make_range(block_range, a.threads)
  for i, irange in enumerate(range_list):
      ptask.append(TaskIterator(irange, i*odd_even))
  return ptask

def make_render_tasks(a, block_offset, block_range, block_types, dsts,
                      cm, src, serial_field, bbox, mip, src_mask_cv,
                      src_mask_val, src_mask_mip):
  class TaskIterator(object):
      def __init__(self, brange, odd_even):
          self.brange = brange
          self.odd_even = odd_even
      def __iter__(self):
          prefix = block_offset
          for i, block_start in enumerate(self.brange):
              block_type = block_types[(i+self.odd_even) % 2]
              dst = dsts[block_type]
              z = block_start + block_offset
              t = a.render(cm, src, serial_field, dst, src_z=z, field_z=z, dst_z=z,
                           bbox=bbox, src_mip=mip, field_mip=mip, mask_cv=src_mask_cv,
                           mask_val=src_mask_val, mask_mip=src_mask_mip, prefix=prefix)
              yield from t
  ptask = []
  range_list, odd_even = make_range(block_range, a.threads)
  for i, irange in enumerate(range_list):
      ptask.append(TaskIterator(irange, i*odd_even))
  return ptask

def make_vvote_tasks(a, block_range, block_offset, cm, pair_fields,
                     vvote_field, bbox, mip):
  class TaskIterator(object):
      def __init__(self, brange):
          self.brange = brange
      def __iter__(self):
          prefix = block_offset
          for block_start in self.brange:
              z = block_start + block_offset
              t = a.vector_vote(cm, pair_fields, vvote_field, z, bbox,
                                mip, inverse=False, serial=True, prefix=prefix)
              yield from t
  ptask = []
  range_list, odd_even = make_range(block_range, a.threads)
  for irange in range_list:
      ptask.append(TaskIterator(irange))
  return ptask

