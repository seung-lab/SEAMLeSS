import concurrent.futures
from copy import deepcopy, copy
from functools import partial
import json
import math
import os
from os.path import join
from time import time, sleep
from torch.autograd import Variable
from multiprocessing import Process, Manager

from pathos.multiprocessing import ProcessPool, ThreadPool
from threading import Lock

import csv
from cloudvolume import Storage
from cloudvolume.lib import Vec
import numpy as np
import scipy
import scipy.ndimage
from skimage.morphology import disk as skdisk
from skimage.filters.rank import maximum as skmaximum 
from taskqueue import TaskQueue, LocalTaskQueue
import torch
from torch.nn.functional import interpolate
import torch.nn as nn

from normalizer import Normalizer
from scipy.special import binom
from temporal_regularization import create_field_bump
from utilities.helpers import save_chunk, crop, upsample, grid_sample, \
                              np_downsample, invert, compose_fields, upsample_field, \
                              is_identity, cpc, vector_vote, get_affine_field, is_blank, \
                              identity_grid
from boundingbox import BoundingBox, deserialize_bbox

from pathlib import Path
from utilities.archive import ModelArchive
from utilities.helpers import vvmodel, warp_model

import torch.nn as nn
from taskqueue import TaskQueue
import tasks
import tenacity
import boto3
from fcorr import get_fft_power2, get_hp_fcorr

retry = tenacity.retry(
  reraise=True, 
  stop=tenacity.stop_after_attempt(7), 
  wait=tenacity.wait_full_jitter(0.5, 60.0),
)

class Aligner:
  def __init__(self, threads=1, queue_name=None, task_batch_size=1, 
                     dry_run=False, **kwargs):
    print('Creating Aligner object')

    self.distributed = (queue_name != None)
    self.queue_name = queue_name
    self.task_queue = None
    self.sqs = None
    self.queue_url = None
    if queue_name:
      self.task_queue = TaskQueue(queue_name=queue_name, n_threads=0)
    self.chunk_size = (1024, 1024)
    self.device = torch.device('cuda')

    self.model_archives = {}
    self.vvmodel = vvmodel()
    self.warp_model = warp_model()
 
    # self.pool = None #ThreadPool(threads)
    self.threads = threads
    self.task_batch_size = task_batch_size
    self.dry_run = dry_run
    self.eps = 1e-6

    #self.gpu_lock = kwargs.get('gpu_lock', None)  # multiprocessing.Semaphore

  def convert_to_float(self, data):
      data = data.type(torch.float)
      data = data / 255.0
      return data

  def convert_to_uint8(self, data):
      data = data * 255
      data = data.type(torch.uint8)
      return data

  def convert_to_int16(self, data):
      if(torch.max(data) > 8192 or torch.min(data)< -8192):
          print('Value in field is out of range of int16 max: {}, min: {}'.format(
                                               torch.max(data), torch.min(data)), flush=True)
      data = data * 4
      return data.type(torch.int16)

  def int16_to_float(self, data):
      return data.type(torch.float)/4.0

  def new_align_task(self, z_range, src, dst, s_field, vvote_field, chunk_grid, mip,
                     pad, radius, block_size, chunk_size, model_lookup,
                     src_mask_cv=None, src_mask_mip=0, src_mask_val=0, rows=1000,
                     super_chunk_len=1000, overlap_chunks=0):
      #print("---------------------------->> load image")
      batch = []
      for i in z_range:
          batch.append(tasks.NewAlignTask(src, dst, s_field, vvote_field, chunk_grid, mip, pad,
                                          radius, i, block_size, chunk_size,
                                          model_lookup, src_mask_cv, src_mask_mip,
                                          src_mask_val, rows, super_chunk_len,
                                          overlap_chunks))
      return batch


  def new_align(self, src, dst, s_field, vvote_field, chunk_grid, mip, pad, radius, block_start,
                block_size, chunk_size, lookup_path, src_mask_cv=None, src_mask_mip=0,
                src_mask_val=0, rows=1000, super_chunk_len=1000, overlap_chunks=0):
      model_lookup={}
      with open(lookup_path) as f:
        reader = csv.reader(f, delimiter=',')
        for k, r in enumerate(reader):
           if k != 0:
             x_start = int(r[0])
             y_start = int(r[1])
             z_start = int(r[2])
             x_stop  = int(r[3])
             y_stop  = int(r[4])
             z_stop  = int(r[5])
             bbox_mip = int(r[6])
             model_path = join('..', 'models', r[7])
             #bbox = BoundingBox(x_start, x_stop, y_start, y_stop, bbox_mip, max_mip)
             for z in range(z_start, z_stop):
               #bbox_lookup[z] = bbox 
               model_lookup[z] = model_path

      overlap = radius
      full_range = range(block_size + overlap)
      vvote_range = full_range[overlap:]

      copy_range = full_range[overlap-1:overlap]
      serial_range = full_range[:overlap-1][::-1]

      serial_offsets = {serial_range[i]: i+1 for i in range(overlap-1)}
      vvote_offsets = [-i for i in range(1, overlap+1)]

      #chunk_grid = self.get_chunk_grid(cm, bbox, mip, 0, rows, pad)
      print("copy range ", copy_range)
      #print("---- len of chunks", len(chunk_grid), "orginal bbox", bbox.stringify(0))
      vvote_range_small = vvote_range
      #vvote_range_small = vvote_range[:super_chunk_len-overlap+1]
      #vvote_subblock = range(vvote_range_small[-1]+1, vvote_range[-1]+1,
      #                          super_chunk_len)
      print("--------overlap is ", overlap, "vvote_range_small",
            vvote_range_small)
      #dst = dsts[0]
      #for i in vvote_subblock:
      #    print("*****>>> vvote subblock is ", i)
      #for i in  chunk_grid:
      #    print("--------grid size is ", i.stringify(0, mip=mip))
      for i in range(len(chunk_grid)):
          chunk = chunk_grid[i]
          if(len(chunk_grid) == 1):
              head_crop = False
              end_crop = False
          else:
              if i == 0:
                  head_crop = False
                  end_crop = True
              elif i == (len(chunk_grid) -1):
                  head_crop = True
                  end_crop = False
              else:
                  head_crop = True
                  end_crop = True

          if head_crop == False and end_crop == True:
              final_chunk = self.crop_chunk(chunk, mip, pad,
                                         chunk_size*(super_chunk_len-1)+pad,
                                         pad, pad)
          elif head_crop and end_crop:
              final_chunk = self.crop_chunk(chunk, mip, chunk_size*(super_chunk_len-1)+pad,
                                         chunk_size*(super_chunk_len-1)+pad,
                                         pad, pad)
          elif head_crop and end_crop == False:
              final_chunk = self.crop_chunk(chunk, mip, chunk_size*(super_chunk_len-1)+pad,
                                         pad, pad, pad)
          else:
              final_chunk = self.crop_chunk(chunk, mip, pad,
                                         pad, pad, pad)
          print("----------------------- head crop ", head_crop, " end_crop",
                end_crop)

          print("<<<<<<init chunk size is ", chunk.stringify(0, mip=mip),
                "final_chunk is ", final_chunk.stringify(0, mip=mip))
          image_list, chunk = self.process_super_chunk_serial(src, block_start, copy_range[0],
                                                              serial_range, serial_offsets,
                                                              s_field, model_lookup,
                                                              chunk, mip, pad, chunk_size,
                                                              head_crop, end_crop,
                                                              final_chunk,
                                                              mask_cv=src_mask_cv,
                                                              mask_mip=src_mask_mip,
                                                              mask_val=src_mask_val)
          write_image = []
          for _ in image_list:
              write_image.append(True)
          print("============================ start vector voting")
          # align with vector voting
          vvote_way = radius
          self.process_super_chunk_vvote(src, block_start, vvote_range_small, dst,
                                      model_lookup, vvote_way, image_list,
                                      write_image, chunk,
                                      mip, pad, chunk_size, super_chunk_len,
                                      vvote_field,
                                      head_crop, end_crop, final_chunk,
                                      mask_cv=src_mask_cv, mask_mip=src_mask_mip,
                                      mask_val=src_mask_val)

  def mp_compute_field_singel(self, dic, coor_list, device_num,
                              chunk_size, pad, padded_len):
      start = time()
      ppid = os.getpid()
      print("start a new process to compute vector field", ppid)
      src_img = dic['src']
      tgt_img = dic['tgt']
      dst_field = dic['field']
      for xs, ys in coor_list:
          src_patch = src_img[...,xs*chunk_size:xs*chunk_size+padded_len,
                            ys*chunk_size:ys*chunk_size+padded_len]
          tgt_patch = tgt_img[...,xs*chunk_size:xs*chunk_size+padded_len,
                            ys*chunk_size:ys*chunk_size+padded_len]

          src_patch = src_patch.to(device='cuda:'+device_num)
          tgt_patch = tgt_patch.to(device='cuda:'+device_num)
          src_patch = self.convert_to_float(src_patch)
          tgt_patch = self.convert_to_float(tgt_patch)
          field = self.new_compute_field_chunk(model_path, src_patch,
                                           tgt_patch, False)
          field = field[:,pad:-pad,pad:-pad,:]
          field = field.to(device='cpu')
          dst_field[:,pad+xs*chunk_size:pad+xs*chunk_size+chunk_size,
                    pad+ys*chunk_size:pad+ys*chunk_size+chunk_size,:] = field

      print("end of the computing vector process {}, use {}".format(ppid,
                                                                    time()-start,
                                                                    flush=True))

  def mp_compute_field_multi_GPU(self, model_path, src_img, tgt_img, chunk_size, pad,
                        warp=False):
      #print("--------------- src and tgt shape", src_img.shape, tgt_img.shape)
      img_shape = src_img.shape
      x_len = img_shape[-2]
      y_len = img_shape[-1]
      unpadded_size = x_len - 2*pad
      padded_len = chunk_size + 2*pad
      y_chunk_number = (y_len - 2*pad) // chunk_size
      x_chunk_number = unpadded_size // chunk_size
      dst_field = torch.FloatTensor(1, x_len, y_len, 2).zero_()
      #print("--------------IN compute field", x_chunk_number*y_chunk_number)
      coor_list = []
      for i in range(x_chunk_number):
          for j in range(y_chunk_number):
              coor_list.apend((i, j))
      gpu_num = torch.cuda.device_count()
      part_len = round(len(coor_list)/gpu_num)
      p_list = []
      m = Manager()
      dic = m.dict()
      dic['src'] = src_img
      dic['tgt'] = tgt_img
      dic['field'] = dst_field
      for i in range(gpu_num):
          if(i==len(gpu_num)-1):
              p = Process(target=self.mp_compute_field_singel,
                          args=(dic, coor_list[i*part_len:], i, chunk_size, pad,
                                padded_len))
          else:
              p = Process(target=self.mp_compute_field_singel,
                          args=(dic, coor_list[i*part_len:(i+1)*part_len], i,
                                chunk_size, pad, padded_len))
          p.start()
          p_list.append(p)
      for i in p_list:
          i.join()
      if(warp):
          get_corr = coor(x_chunk_number, y_chunk_number)
          image = self.warp_slice(chunk_size, pad, src_img, dst_field,
                                  get_corr)
          return image, dst_field
      else:
          return dst_field


  def new_compute_field_multi_GPU(self, model_path, src_img, tgt_img, chunk_size, pad,
                        warp=False):
      #print("--------------- src and tgt shape", src_img.shape, tgt_img.shape)
      img_shape = src_img.shape
      x_len = img_shape[-2]
      y_len = img_shape[-1]
      unpadded_size = x_len - 2*pad
      padded_len = chunk_size + 2*pad
      y_chunk_number = (y_len - 2*pad) // chunk_size
      x_chunk_number = unpadded_size // chunk_size
      #if(warp):
      #    #if(first_chunk):
      #    #    adjust = pad
      #    #else:
      #    #    adjust = 0
      #    image = torch.FloatTensor(1, 1, unpadded_size, y_len).zero_()
      #else:
      #dst_field = torch.FloatTensor(1, unpadded_size, y_len, 2).zero_()
      dst_field = torch.FloatTensor(1, x_len, y_len, 2).zero_()
      #print("--------------IN compute field", x_chunk_number*y_chunk_number)
      def coor(x,y):
          for i in range(x):
              for j in range(y):
                  yield i ,j
          yield -1,-1
      gpu_num = torch.cuda.device_count()
      get_corr = coor(x_chunk_number, y_chunk_number)
      has_next = True
      while(has_next):
          coor_list = []
          xs, ys = next(get_corr)
          if(xs ==-1):
              break
          src_patch = src_img[...,xs*chunk_size:xs*chunk_size+padded_len,
                            ys*chunk_size:ys*chunk_size+padded_len]
          tgt_patch = tgt_img[...,xs*chunk_size:xs*chunk_size+padded_len,
                            ys*chunk_size:ys*chunk_size+padded_len]
          coor_list.append([xs, ys])
          for i in range(1, gpu_num):
              xs, ys = next(get_corr)
              if(xs == -1):
                  has_next = False
                  break
              else:
                  tmp = src_img[...,xs*chunk_size:xs*chunk_size+padded_len,
                                  ys*chunk_size:ys*chunk_size+padded_len]
                  src_patch = torch.cat((src_patch, tmp),0)
                  tgt_tmp = tgt_img[...,xs*chunk_size:xs*chunk_size+padded_len,
                            ys*chunk_size:ys*chunk_size+padded_len]
                  tgt_patch = torch.cat((tgt_patch, tgt_tmp),0)
                  coor_list.append([xs, ys])
          src_patch = src_patch.to(device=self.device)
          tgt_patch = tgt_patch.to(device=self.device)
          src_patch = self.convert_to_float(src_patch)
          tgt_patch = self.convert_to_float(tgt_patch)

          field = self.new_compute_field_chunk(model_path, src_patch,
                                           tgt_patch, False)
          field = field[:,pad:-pad,pad:-pad,:]
          field = field.to(device='cpu')
          for i in range(len(coor_list)):
              xs, ys = coor_list[i]
              dst_field[:,pad+xs*chunk_size:pad+xs*chunk_size+chunk_size,
                    pad+ys*chunk_size:pad+ys*chunk_size+chunk_size,:] = field[i:i+1,...]
      if(warp):
          get_corr = coor(x_chunk_number, y_chunk_number)
          image = self.warp_slice(chunk_size, pad, src_img, dst_field,
                                  get_corr)
          return image, dst_field
      else:
          return dst_field

          #if warp:
          #    image_patch = self.new_compute_field_chunk(model_path, src_patch,
          #                                              tgt_patch, warp, pad)
          #    for i in range(len(coor_list)):
          #        xs, ys = coor_list[i]
          #        image[...,xs*chunk_size:xs*chunk_size+chunk_size,
          #            pad+ys*chunk_size:pad+ys*chunk_size+chunk_size] = image_patch[i]
          #   #del image_patch
          #else:
          #    field = self.new_compute_field_chunk(model_path, src_patch,
          #                                     tgt_patch, warp)
          #    field = field[:,pad:-pad,pad:-pad,:]
          #    field = field.to(device='cpu')
          #    for i in range(len(coor_list)):
          #        xs, ys = coor_list[i]
          #        dst_field[:,xs*chunk_size:xs*chunk_size+chunk_size,
          #              pad+ys*chunk_size:pad+ys*chunk_size+chunk_size,:] = field[i:i+1,...]

  def warp_slice(self, chunk_size, pad, src_img, dst_field, get_corr):
      img_shape = src_img.shape
      x_len = img_shape[-2]
      y_len = img_shape[-1]
      padded_len = chunk_size + 2*pad
      x_chunk_number = (x_len - 2*pad) // chunk_size
      y_chunk_number = (y_len - 2*pad) // chunk_size

      image = torch.FloatTensor(1, 1, x_len-2*pad, y_len).zero_()
      has_next = True
      #gpu_num = torch.cuda.device_count()
      gpu_num = 1
      while(has_next):
          coor_list = []
          xs, ys = next(get_corr)
          if(xs ==-1):
              break
          field = dst_field[:,xs*chunk_size:xs*chunk_size+padded_len,
                            ys*chunk_size:ys*chunk_size+padded_len,:]
          distance = self.profile_field(field).type(torch.int32)
          field -= distance.type(torch.float32)
          dis = distance.flip(0)
          src_patch = src_img[...,dis[0]+xs*chunk_size:dis[0]+xs*chunk_size+padded_len,
                                  dis[1]+ys*chunk_size:dis[1]+ys*chunk_size+padded_len]
          coor_list.append([xs, ys])
          for i in range(1, gpu_num):
              xs, ys = next(get_corr)
              if(xs == -1):
                  has_next = False
                  break
              else:
                  tmp_f = dst_field[:,xs*chunk_size:xs*chunk_size+padded_len,
                            ys*chunk_size:ys*chunk_size+padded_len,:]
                  distance = self.profile_field(tmp_f).type(torch.int32)
                  tmp_f -= distance.type(torch.float32)
                  field = torch.cat((field, tmp_f),0)
                  dis = distance.flip(0)
                  tmp_s = src_img[...,dis[0]+xs*chunk_size:dis[0]+xs*chunk_size+padded_len,
                                  dis[1]+ys*chunk_size:dis[1]+ys*chunk_size+padded_len]
                  src_patch = torch.cat((src_patch, tmp_s),0)
                  coor_list.append([xs, ys])
          src_patch = src_patch.to(device=self.device)
          src_patch = self.convert_to_float(src_patch)
          field = field.to(device=self.device)
          image_patch = self.new_cloudsample_image(src_patch, field)
          image_patch = image_patch[:,:,pad:-pad,pad:-pad]
          image_patch = image_patch.to(device='cpu')
          for i in range(len(coor_list)):
              xs, ys = coor_list[i]
              image[..., xs*chunk_size:xs*chunk_size+chunk_size,
                    pad+ys*chunk_size:pad+ys*chunk_size+chunk_size]=image_patch[i:i+1,...]
      return image


  def new_compute_field(self, model_path, src_img, tgt_img, chunk_size, pad,
                        warp=False):
      #print("--------------- src and tgt shape", src_img.shape, tgt_img.shape)
      img_shape = src_img.shape
      x_len = img_shape[-2]
      y_len = img_shape[-1]
      unpadded_size = x_len - 2*pad
      padded_len = chunk_size + 2*pad
      y_chunk_number = (y_len - 2*pad) // chunk_size
      x_chunk_number = unpadded_size // chunk_size
      if(warp):
          #if(first_chunk):
          #    adjust = pad
          #else:
          #    adjust = 0
          image = torch.FloatTensor(1, 1, unpadded_size, y_len).zero_()
      else:
          dst_field = torch.FloatTensor(1, unpadded_size, y_len, 2).zero_()
      #print("--------------IN compute field", x_chunk_number*y_chunk_number)
      for xs in range(x_chunk_number):
          for ys in range(y_chunk_number):
              src_patch = src_img[...,xs*chunk_size:xs*chunk_size+padded_len,
                                  ys*chunk_size:ys*chunk_size+padded_len]
              tgt_patch = tgt_img[...,xs*chunk_size:xs*chunk_size+padded_len,
                                  ys*chunk_size:ys*chunk_size+padded_len]
              src_patch = src_patch.to(device=self.device)
              tgt_patch = tgt_patch.to(device=self.device)
              src_patch = self.convert_to_float(src_patch)
              #if tgt_patch.dtype == torch.uint8:
              #    tgt_patch = self.convert_to_float(tgt_patch)
              start_t =  time()
              if warp:
                  image_patch = self.new_compute_field_chunk(model_path, src_patch,
                                                   tgt_patch, warp)
                  image_patch = image_patch[:,:,pad:-pad,pad:-pad]
                  #image_patch = self.convert_to_uint8(image_patch)
                  image_patch = image_patch.to(device='cpu')
                  image[...,xs*chunk_size:xs*chunk_size+chunk_size,
                       pad+ys*chunk_size:pad+ys*chunk_size+chunk_size] = image_patch
              else:
                  field = self.new_compute_field_chunk(model_path, src_patch,
                                                   tgt_patch, warp)
                  field = field[:,pad:-pad,pad:-pad,:]
                  field = field.to(device='cpu')
                  dst_field[:,xs*chunk_size:xs*chunk_size+chunk_size,
                            pad+ys*chunk_size:pad+ys*chunk_size+chunk_size,:] = field
              end_t = time()
              #print("------------------compute field time", end_t - start_t)

      #for xs in [x_chunk_number-1]:
      #    for ys in range(y_chunk_number):
      #        src_patch = src_img[...,xs*chunk_size:xs*chunk_size+padded_len,
      #                            ys*chunk_size:ys*chunk_size+padded_len]
      #        tgt_patch = tgt_img[...,xs*chunk_size:xs*chunk_size+padded_len,
      #                            ys*chunk_size:ys*chunk_size+padded_len]
      #        src_patch = src_patch.to(device=self.device)
      #        tgt_patch = tgt_patch.to(device=self.device)
      #        src_patch = self.convert_to_float(src_patch)
      #        tgt_patch = self.convert_to_float(tgt_patch)
      #        if warp:
      #            image_patch = self.new_compute_field_chunk(model_path, src_patch,
      #                                             tgt_patch, warp)
      #            image_patch = image_patch[:,:,pad:-pad,pad:-pad]
      #            image_patch = self.convert_to_uint8(image_patch)
      #            image_patch = image_patch.to(device='cpu')
      #            image[...,xs*chunk_size:xs*chunk_size+chunk_size,
      #                 pad+ys*chunk_size:pad+ys*chunk_size+chunk_size] = image_patch
      #        else:
      #            field = self.new_compute_field_chunk(model_path, src_patch,
      #                                             tgt_patch, warp) 
      #            field = field[:,pad:-pad,pad:-pad,:]
      #            field = field.to(device='cpu')
      #            dst_field[:,xs*chunk_size:xs*chunk_size+chunk_size,
      #                      ys*chunk_size:ys*chunk_size+chunk_size,:] = field

      if(warp):
          return image
      else:
          return dst_field

  def crop_imageX(self, image, head_crop, end_crop, amount):
      if amount == 0:
          return image
      head_crop_len = amount if head_crop else 0
      croped_image = image[..., head_crop_len:-amount,:] if end_crop else image[..., head_crop_len:,:]
      return croped_image

  def get_section_field(self, src, block_start, copy_offset, serial_range,
                        serial_offsets, field_cv0, field_cv1, model_lookup,
                        schunk, mip, pad, chunk_size,
                        head_crop, end_crop, final_chunk):
      #image_list = []
      #load from copy range
      chunk = deepcopy(schunk)
      print("---- chunk is ", chunk.stringify(0, mip=mip), " z is",
            block_start+copy_offset)
      load_image_start = time()
      tgt_image = self.load_part_image(src, block_start+copy_offset,
                                  chunk, mip)
      load_finish = time()
      print("----------------LOAD image time:", load_finish-load_image_start)
      #tgt_image = self.convert_to_float(tgt_image)
      crop_len = chunk_size*copy_offset
      add_image =self.crop_imageX(tgt_image, head_crop, end_crop, crop_len)
      #image_list.append(add_image)

      #for block_offset in serial_range:
      block_offset = serial_range[0]
      z_offset = serial_offsets[block_offset]
      #dst = dsts[even_odd]
      z = block_start + block_offset
      print("---------------- z ", z, "  block_offset ", block_offset)
      model_path = model_lookup[z]
      load_image_start = time()
      src_image = self.load_part_image(src, z, chunk, mip)
      #print("++++++chunk is", chunk.stringify(0, mip=mip), "src_image shape",
      #                                  src_image.shape, "tgt_image",
      #                                  tgt_image.shape)
      load_finish = time()
      print("----------------LOAD image time:", load_finish-load_image_start)
      dst_field = self.new_compute_field_multi_GPU(model_path, src_image, tgt_image,
                                      chunk_size, pad, warp=False)
      print("----------------COMPUTE FIELD time", time()- load_finish)
      dst_field = dst_field[...,pad:-pad,:].data.cpu().numpy()
      print("++ dst_field shape", dst_field.shape, "type", type(dst_field.shape))
      store_field_compress = time() 
      self.save_field(dst_field, field_cv0, z, final_chunk, mip, relative=False,
                       as_int16=True)
      store_field_end = time() 
      print("----------------store compressed field time:",
            store_field_end-store_field_compress)
      self.save_field(dst_field, field_cv1, z, final_chunk, mip, relative=False,
                       as_int16=True)
      print("----------------store uncompressd field time:",
            time()-store_field_end)
      #chunk = a.adjust_chunk(chunk, mip, chunk_size, first_chunk=first_chunk)
      #return image_list, chunk

  def get_aligned_section(self, src, block_start, copy_offset, serial_range,
                        serial_offsets, field_cv0, field_cv1, model_lookup,
                        schunk, mip, pad, chunk_size,
                        head_crop, end_crop, final_chunk):
      #image_list = []
      #load from copy range
      chunk = deepcopy(schunk)
      print("---- chunk is ", chunk.stringify(0, mip=mip), " z is",
            block_start+copy_offset)
      load_image_start = time()
      tgt_image = self.load_part_image(src, block_start+copy_offset,
                                  chunk, mip)
      load_finish = time()
      print("----------------LOAD image time:", load_finish-load_image_start)
      #tgt_image = self.convert_to_float(tgt_image)
      crop_len = chunk_size*copy_offset
      add_image =self.crop_imageX(tgt_image, head_crop, end_crop, crop_len)
      #image_list.append(add_image)

      #for block_offset in serial_range:
      block_offset = serial_range[0]
      z_offset = serial_offsets[block_offset]
      #dst = dsts[even_odd]
      z = block_start + block_offset
      print("---------------- z ", z, "  block_offset ", block_offset)
      model_path = model_lookup[z]
      load_image_start = time()
      src_image = self.load_part_image(src, z, chunk, mip)
      #print("++++++chunk is", chunk.stringify(0, mip=mip), "src_image shape",
      #                                  src_image.shape, "tgt_image",
      #                                  tgt_image.shape)
      load_finish = time()
      print("----------------LOAD image time:", load_finish-load_image_start)
      image = self.new_compute_field_multi_GPU(model_path, src_image, tgt_image,
                                      chunk_size, pad, warp=True)
      print("----------------COMPUTE FIELD time and aligned: ", time()- load_finish)
      #dst_field = dst_field[...,pad:-pad,:].data.cpu().numpy()
      #print("++ dst_field shape", dst_field.shape, "type", type(dst_field.shape))
      #store_field_compress = time() 
      #self.save_field(dst_field, field_cv0, z, final_chunk, mip, relative=False,
      #                 as_int16=True)
      #store_field_end = time() 
      #print("----------------store compressed field time:",
      #      store_field_end-store_field_compress)
      #self.save_field(dst_field, field_cv1, z, final_chunk, mip, relative=False,
      #                 as_int16=True)
      #print("----------------store uncompressd field time:",
      #      time()-store_field_end)
      #chunk = a.adjust_chunk(chunk, mip, chunk_size, first_chunk=first_chunk)
      #return image_list, chunk
  
  def mp_load(self, dic, key, src, z, chunk, mip, mask_cv, mask_mip, mask_val):
      ppid = os.getpid()
      print("start a new process", ppid)
      img = self.load_part_image(src, z, chunk, mip, mask_cv=mask_cv,
                                     mask_mip=mask_mip, mask_val=mask_val)
      #if img.dtype == torch.uint8:
      #    print("need to convert {}".format(os.getpid()), flush=True)
      #    img = self.convert_to_float(img)
      dic[key] = img
      print("end of the new process {}".format(ppid), flush=True)

  def process_super_chunk_serial(self, src, block_start, copy_offset, serial_range,
                                 serial_offsets, dsts, model_lookup,
                                 schunk, mip, pad, chunk_size,
                                 head_crop, end_crop, final_chunk,
                                 mask_cv=None, mask_mip=0,
                                 mask_val=0, affine=None):
      image_list = []
      upload_list = []
      p_list = []
      #load from copy range
      chunk = deepcopy(schunk)
      print("---- chunk is ", chunk.stringify(0, mip=mip), " z is",
            block_start+copy_offset)
      #load_image_start = time()
      m = Manager()
      dic = m.dict()
      tgt_image = self.load_part_image(src, block_start+copy_offset,
                                  chunk, mip, mask_cv=mask_cv,
                                  mask_mip=mask_mip, mask_val=mask_val)
      #load_finish = time()
      #print("----------------LOAD image time:", load_finish-load_image_start)
      #tgt_image = self.convert_to_float(tgt_image) 
      crop_len = chunk_size*copy_offset
      add_image =self.crop_imageX(tgt_image, head_crop, end_crop, crop_len)
      image_list.append(add_image)
      src_image0 = self.load_part_image(src,block_start+serial_offsets[0], chunk, mip,
                                        mask_cv=mask_cv, mask_mip=mask_mip,
                                        mask_val=mask_val)
      head_crop_len = chunk_size if head_crop else 0
      end_crop_len = chunk_size if end_crop else 0
      chunk= self.crop_chunk(chunk, mip, head_crop_len, end_crop_len, 0,
                            0)
      for block_offset in serial_range:
          z_offset = serial_offsets[block_offset]
          #serial_field = serial_fields[z_offset]
          #dst = dsts[even_odd]
          dst = dsts
          z = block_start + block_offset
          print("---------------- z ", z, "  block_offset ", block_offset)
          model_path = model_lookup[z]
          #load_image_start = time()
          #src_image = self.load_part_image(src, z, chunk, mip, mask_cv=mask_cv,
          #                            mask_mip=mask_mip,
          #                            mask_val=mask_val)
          if block_offset == serial_range[0]:
              src_image= src_image0
          else:
              for i in p_list:
                  i.join()
              #print(p_list)
              #print("len of dic", len(dic))
              src_image = dic["src"]
              del dic["src"]
          if block_offset != serial_range[-1]:
              p_list = []
              #self.mp_load(dic,"src", src, z-1, chunk,
              #          mip, mask_cv, mask_mip,
              #          mask_v)
              p = Process(target=self.mp_load, args=(dic, "src", src, z-1, chunk,
                                                     mip, mask_cv, mask_mip,
                                                     mask_val))
              p.start()
              p_list.append(p)
          #print("++++++chunk is", chunk.stringify(0, mip=mip), "src_image shape",
          #                                  src_image.shape, "tgt_image",
          #                                  tgt_image.shape)
          load_finish = time()
          #print("----------------LOAD image time:", load_finish-load_image_start)
          new_tgt_image, dst_field = self.new_compute_field_multi_GPU(model_path, src_image,
                                                                      tgt_image, chunk_size,
                                                                      pad, warp=True)
          print("----------------COMPUTE FIELD and warp time", time()-load_finish)
          if(not head_crop):
              pad_tensor = torch.FloatTensor(1, 1, pad, new_tgt_image.shape[-1]).zero_()
              new_tgt_image = torch.cat((pad_tensor, new_tgt_image), 2)
          if(not end_crop):
              pad_tensor = torch.FloatTensor(1, 1, pad,new_tgt_image.shape[-1]).zero_()
              new_tgt_image = torch.cat((new_tgt_image,pad_tensor), 2)
          image_crop_len = chunk_size*block_offset+(chunk_size-pad)
          tgt_crop_len = chunk_size-pad
          croped_image = self.crop_imageX(new_tgt_image, head_crop, end_crop,
                                         image_crop_len)
          image_list.insert(0, croped_image)
          #Algin frist several slices to anchor
          #new_tgt_image = self.crop_imageX(new_tgt_image, head_crop, end_crop,
          #                            tgt_crop_len)
          #print("------------tgt_image shape", tgt_image.shape)
          #adjusted_box= a.adjust_chunk(chunk, mip,
          #                             chunk_size*block_offset+chunk_size, first_chunk)
          #print("-----------------adjusted_box", adjusted_box.stringify(0,
          #                                                              mip=mip),
          #     "chunk is", chunk.stringify(0, mip=mip))
          #bbox_list.insert(0, adjusted_box)
          #head_crop_len = chunk_size if head_crop else 0
          #end_crop_len = chunk_size if end_crop else 0
          chunk= self.crop_chunk(chunk, mip, head_crop_len, end_crop_len, 0,
                                 0)
          #chunk = a.adjust_chunk(chunk, mip, chunk_size, first_chunk=first_chunk)
          for i in upload_list:
              i.join()
          upload_list = []
          dic["field"] =dst_field
          # set x_range_len = None since align whole images. No head_crop and no
          # end_crop
          x_range_len = None
          pf = Process(target=self.mp_store_field, args=(dic,"field", head_crop,
                                                        end_crop, x_range_len,
                                                        pad, dst, z,
                                                        final_chunk, mip,
                                                        chunk_size))
          pf.start()
          upload_list.append(pf)
      return image_list, chunk

  def mp_store_img(self, dic, key, dst, z, chunk, mip, to_uint8):
      ppid = os.getpid()
      print("start a new save image process", ppid)
      image = dic[key]
      self.save_image(image.cpu().numpy(), dst, z, chunk, mip, to_uint8=to_uint8)
      del dic[key]
      print("end of the save image process {}".format(ppid), flush=True)

  def mp_store_field(self, dic, key, head_crop, end_crop, x_range_len, pad, dst, z,
                     chunk, mip, chunk_size):
      ppid = os.getpid()
      print("start a new save field process", ppid)
      field = dic[key]
      field_len = field.shape[1] - 2*pad
      if(head_crop and end_crop):
          crop_amount = (field_len - x_range_len)/2
          field = field[:,pad+crop_amount:-(pad+crop_amount),pad:-pad,:]
      elif head_crop:
          crop_amount = (field_len - x_range_len)
          field = field[:,pad+crop_amount:-pad,pad:-pad,:]
      elif end_crop:
          crop_amount = (field_len - x_range_len)
          field = field[:,pad:-(pad+crop_amount),pad:-pad,:]
      else:
          field = field[:,pad:-pad,pad:-pad,:]
      print("***********dst_field shape", field.shape)
      #field_from_GPU = time()
      field = field.cpu().numpy() * ((chunk_size+2*pad)/ 2) * (2**mip)
      field_on_CPU = time()
      #print("-----------------move field from GPU to CPU time",
      #      field_on_CPU-field_from_GPU)
      self.save_field(field, dst, z, chunk, mip, relative=False,
                   as_int16=True)
      print("-------------------Saving field time:", time()-field_on_CPU)
      del dic[key]
      print("end of the save field process {}".format(ppid), flush=True)


  def process_super_chunk_vvote(self, src, block_start, vvote_range_small,
                                dsts, model_lookup, vvote_way, image_list,
                                write_image,
                                schunk, mip, pad, chunk_size, super_chunk_len,
                                vvote_field,
                                head_crop, end_crop, final_chunk, mask_cv=None,
                                mask_mip=0, mask_val=0):
      chunk = deepcopy(schunk)
      m = Manager()
      dic = m.dict()
      p_list = []
      upload_list = []
      src_image0 = self.load_part_image(src, block_start+vvote_range_small[0], chunk,
                                        mip, mask_cv=mask_cv, mask_mip=mask_mip,
                                        mask_val=mask_val)
      head_crop_len = chunk_size if head_crop else 0
      end_crop_len = chunk_size if end_crop else 0
      chunk= self.crop_chunk(chunk, mip, head_crop_len, end_crop_len, 0, 0)
      for block_offset in vvote_range_small:
          print("===========block offset is", block_offset)
          whole_vv_start = time()
          dst = dsts
          z = block_start + block_offset
          model_path = model_lookup[z]
          #vvote_way = args.tgt_radius
          if block_offset == vvote_range_small[0]:
              src_image= src_image0
          else:
              for i in p_list:
                  i.join()
              #print(p_list)
              #print("len of dic", len(dic))
              src_image = dic["src"]
              del dic["src"]
          if block_offset != vvote_range_small[-1]:
              p_list = []
              #self.mp_load(dic,"src", src, z-1, chunk,
              #          mip, mask_cv, mask_mip,
              #          mask_val)
              p = Process(target=self.mp_load, args=(dic, "src", src, z+1, chunk,
                                                     mip, mask_cv, mask_mip,
                                                     mask_val))
              p.start()
              p_list.append(p)
          #load_image_start = time()
          #src_image = self.load_part_image(src, z, chunk, mip, mask_cv=mask_cv,
          #                            mask_mip=mask_mip,
          #                            mask_val=mask_val)
          #load_finish = time()
          #print("----------------LOAD image in VV time:", load_finish-load_image_start)
          print("chunk shape for vvoting is", chunk.stringify(0, mip=mip))
          #for i in range(len(image_list)):
          #    print("************shape of image", image_list[i].shape,
          #          image_list[i].dtype)
          print(">--------------------start vvote---------------------->")
          vv_start = time()
          image, dst_field = self.new_vector_vote(model_path, src_image, image_list,
                                                  chunk_size, pad, vvote_way, mip,
                                                  inverse=False, serial=True,
                                                  head_crop=head_crop,
                                                  end_crop=end_crop)
          vv_end =time()
          print("---------------------VV time :", vv_end-vv_start) 
          if(head_crop==False):
              if(block_offset != vvote_range_small[-1]):
                  pad_tensor = torch.FloatTensor(1, 1, pad, image.shape[-1]).zero_()
                  image = torch.cat((pad_tensor, image), 2)
          if(end_crop==False):
              if(block_offset != vvote_range_small[-1]):
                  pad_tensor = torch.FloatTensor(1, 1, pad, image.shape[-1]).zero_()
                  image = torch.cat((image,pad_tensor), 2)
          print("--------------------Padding time", time() - vv_end)

          #image_bbox = a.crop_chunk(bbox_list[0], mip, pad, pad, pad, pad)
          #print("image shape after vvoting", image.shape)
          #del bbox_list[0]
          #print("image_list[0] range ", image_list[0].shape)
          image_len = image_list[0].shape[-2] - 2*pad;
          x_range = final_chunk.x_range(mip=mip)
          x_range_len = x_range[1] - x_range[0]
          print("************ image_list[0]", image_list[0].shape,
                "coresponding_bbx", final_chunk.stringify(0, mip=mip))
          image_chunk =self.crop_imageX(image_list[0][...,pad:-pad], True, True,
                                       pad)

          print("first crop ************ image_chunk", image_chunk.shape)
          if(head_crop and end_crop):
              crop_amount = (image_len - x_range_len)/2
          else:
              crop_amount = (image_len - x_range_len)
          image_chunk =self.crop_imageX(image_chunk, head_crop, end_crop,
                                       crop_amount)
          print("************ image_chunk", image_chunk.shape,
                "coresponding_bbx", final_chunk.stringify(0, mip=mip))
          #IO_start = time()
          for i in upload_list:
              i.join()
          upload_list = []
          if write_image[0]:
              print("write_image len is ", len(write_image), "save image")
              dic["store_img"] = image_chunk
              p = Process(target=self.mp_store_img, args=(dic, "store_img",
                                                          dst,
                                                          z-vvote_way, final_chunk,
                                                          mip, True))
              p.start()
              upload_list.append(p)
              #self.save_image(image_chunk.cpu().numpy(), dst, z-vvote_way,
              #         final_chunk, mip, to_uint8=True)
          #print("------------------- write_image ", write_image[0],
          #      "time",time()-IO_start)
          del image_list[0]
          del write_image[0]
          if(block_offset == vvote_range_small[-1]):
              image = image[...,pad:-pad]
              dic["img"] = image
              pi = Process(target=self.mp_store_img, args=(dic, "img",
                                                          dst, z,
                                                          final_chunk,
                                                          mip, True))
              pi.start()
              upload_list.append(pi)
              #self.save_image(image.cpu().numpy(), dst, z,
              #         final_chunk, mip, to_uint8=True)
          else:
              if(head_crop):
                  image = image[..., chunk_size-pad:,:]
              if(end_crop):
                  image = image[...,0:-(chunk_size-pad),:]
              image_list.append(image)
              write_image.append(True)

          dic["field"] =dst_field
          pf = Process(target=self.mp_store_field, args=(dic,"field", head_crop,
                                                        end_crop, x_range_len,
                                                        pad, vvote_field, z,
                                                        final_chunk, mip,
                                                         chunk_size))
          pf.start()
          upload_list.append(pf)
          chunk= self.crop_chunk(chunk, mip, head_crop_len, end_crop_len, 0,
                                 0)
      print("save image after vvoting --------------->>>>>>>")
      #for i in image_list:
      #    print("************shape of image", i.shape)
      for i in range(len(image_list)):
          print("image shape", image_list[i].shape)
          image_len = image_list[i].shape[-2] - 2*pad;
          x_range = final_chunk.x_range(mip=mip)
          x_range_len = x_range[1] - x_range[0]
          image_chunk = self.crop_imageX(image_list[i][...,pad:-pad], True, True,
                                       pad)
          if(head_crop and end_crop):
              crop_amount = (image_len - x_range_len)/2
          else:
              crop_amount = (image_len - x_range_len)

          print("**** First Crop size ", image_chunk.shape, head_crop, end_crop,
                crop_amount,)
          image_chunk =self.crop_imageX(image_chunk, head_crop, end_crop,
                                       crop_amount)
          #del image_list[0]
          print("************ image_chunk", image_chunk.shape,
                "coresponding_bbx", final_chunk.stringify(0, mip=mip))
          self.save_image(image_chunk.cpu().numpy(), dst, z-(vvote_way-i-1),
                       final_chunk, mip, to_uint8=True)
 
  def new_compute_field_chunk_multi_GPU(self, model_path, src_img, tgt_img, warp=False):
      archive = self.get_model_archive(model_path)
      model = archive.model
      #model = nn.DataParallel(model)
      device_ids= [0,1]
      normalizer = archive.preprocessor
      #if (normalizer is not None): 
      #    if(not is_blank(src_img)):
      #        src_img =normalizer(src_img).reshape(src_img.shape)
      #    if(not is_blank(tgt_img)):
      #        tgt_img =normalizer(tgt_img).reshape(tgt_img.shape)
      #print("***********", src_img.shape, tgt_img.shape, " warp ", warp)
      inputs = torch.cat((src_img, tgt_img),1)
      replicas = nn.parallel.replicate(model, device_ids)
      src = nn.parallel.scatter(inputs, device_ids)
      #torch.cuda.synchronize()
      start_t = time() 
      field = nn.parallel.parallel_apply(replicas, src)
      #field = model(src_img, tgt_img)
      #torch.cuda.synchronize()
      end_t = time()
      #print("+++++++++++++++++compute field time", end_t - start_t)
      if(warp):
          #torch.cuda.synchronize()
          start_t = time() 
          image = self.new_cloudsample_image(src_img, field)
          #torch.cuda.synchronize()
          end_t = time()
          print("+++++++++++++++++ warp time", end_t - start_t)
          return image
      else:
          return field



  def new_compute_field_chunk(self, model_path, src_img, tgt_img, warp=False,
                              pad=None):
      archive = self.get_model_archive(model_path)
      model = archive.model
      model = nn.DataParallel(model).cuda()
      normalizer = archive.preprocessor
      if (normalizer is not None):
          for i in range(src_img.shape[0]):
              if(not is_blank(src_img[i,...])):
                  src_im = src_img[i,...]
                  src_img[i,...] =normalizer(src_im).reshape(src_im.shape)
          for i in range(src_img.shape[0]):
              if(not is_blank(tgt_img[i,...])):
                  tgt_im = tgt_img[i,...]
                  tgt_img[i,...] =normalizer(tgt_im).reshape(tgt_im.shape)
      #torch.cuda.synchronize()
      #start_t = time()
      field = model(src_img, tgt_img)
      #torch.cuda.synchronize()
      #end_t = time()
      #print("+++++++++++++++++compute field time", end_t - start_t, "shape is",
      #     src_img.shape)
      if(warp):
          torch.cuda.synchronize()
          start_t = time()
          image = []
          #print("src_image shape", src_img.shape, "field shape",
          #          field.shape)
          for i in range(src_img.shape[0]):
              im = self.new_cloudsample_image(src_img[i:i+1,...], field[i:i+1,...])
              image.append(im[:,:,pad:-pad,pad:-pad])
          torch.cuda.synchronize()
          end_t = time()
          #print("+++++++++++++++++ warp time", end_t - start_t)
          return image
      else:
          return field

  def adjust_chunk(self, chunk, mip, chunk_size, first_chunk=False):
      if first_chunk:
          start = 0;
      else:
          start= chunk_size;
      x_range = chunk.x_range(mip=mip)
      y_range = chunk.y_range(mip=mip)
      new_chunk = BoundingBox(x_range[0]+start,x_range[1]-chunk_size,
                              y_range[0],y_range[1], mip=mip)
      return new_chunk

  def crop_chunk(self, chunk, mip, x_start, x_end, y_start, y_end):
      x_range = chunk.x_range(mip=mip)
      y_range = chunk.y_range(mip=mip)
      new_chunk = BoundingBox(x_range[0]+x_start, x_range[1]-x_end,
                              y_range[0]+y_start, y_range[1]-y_end, mip=mip)
      return new_chunk

  def random_read(self, cm, image_cv, bbox, mip, pad, z):
      chunk_grid = self.break_into_chunks_grid(bbox, cm.dst_chunk_sizes[mip],
                                          cm.dst_voxel_offsets[mip], mip=mip,
                                          max_mip=cm.max_mip)
      tmp_device = self.device
      self.device = 'cpu'
      chunks = []
      print("***********chunk dim", len(chunk_grid), len(chunk_grid[0]))
      for i in range(len(chunk_grid)):
          chunks.append(BoundingBox(chunk_grid[i][0].x_range(mip=mip)[0]-pad,
                                    chunk_grid[i][-1].x_range(mip=mip)[1]+pad,
                                    chunk_grid[i][0].y_range(mip=mip)[0]-pad,
                                    chunk_grid[i][-1].y_range(mip=mip)[1]+pad,
                                    mip=mip))
      newbb = BoundingBox(chunk_grid[0][0].x_range(mip=mip)[0]-pad,
                          chunk_grid[-1][-1].x_range(mip=mip)[1]+pad,
                          chunk_grid[0][0].y_range(mip=mip)[0]-pad,
                          chunk_grid[-1][-1].y_range(mip=mip)[1]+pad,
                                    mip=mip)
      image = []
      chunk_len = len(chunks)
      odd = chunk_len%2
      start = time()
      im =self.get_image(image_cv, z, newbb, mip, to_tensor=True,
                         normalizer=None, to_float=False)
      #for i in range(chunk_len//2):
      #    image.insert(i, self.get_image(image_cv, z, chunks[i], mip,
      #                                   to_tensor=True, normalizer=None,
      #                                   to_float=False))
      #    image.insert(-i-1, self.get_image(image_cv, z, chunks[-i], mip,
      #                                   to_tensor=True, normalizer=None,
      #                                   to_float=False))
      #if odd:
      #    image.append(self.get_image(image_cv, z, chunks[chunk_len//2+1], mip,
      #                                   to_tensor=True, normalizer=None,
      #                                   to_float=False))
      end =time()
      print("*********random read time:", end-start, "image len", len(image))
      self.device = tmp_device
      print("********** data device", im.device)
      gpu_tensor = torch.randn(im.shape).type(torch.uint8).cuda(0)
      torch.cuda.synchronize()
      start = time()
      gpu_tensor = gpu_tensor.to(device="cpu")
      #new_im = im.to(device="cuda")
      torch.cuda.synchronize()
      end_time = time()
      print("******* move from GPU to cpu ", end_time -start)
      print(gpu_tensor.shape)
      #print("*********move from cpu to GPU", end_time - start)
      del im
      torch.squeeze(new_im, 1)
      new_im += new_im
      new_im += 1
      #new_im = new_im * new_im +3
      torch.cuda.synchronize()
      start = time()
      im = new_im.to(device="cpu")
      torch.cuda.synchronize()
      end_time = time()
      print("******* move from GPU to cpu ", end_time -start)
      del new_im
      im = im + 2
      torch.cuda.synchronize()
      start = time()
      im = im.to(device="cuda")
      torch.cuda.synchronize()
      end_time = time()
      print("******* move from cpu to GUP again ", end_time -start)



      return 0

  def get_chunk_grid(self, cm, bbox, mip, overlap, rows, pad):
      chunk_grid = self.break_into_chunks_grid(bbox, cm.dst_chunk_sizes[mip],
                                          cm.dst_voxel_offsets[mip], mip=mip,
                                          max_mip=cm.max_mip)
      print("--------------chunks_grid shape",len(chunk_grid), len(chunk_grid[0]),
            chunk_grid[0][0].stringify(0))
      chunks = []
      #for i in range(len(chunk_grid)):
      #    for j in range(len(chunk_grid[0])):
      #        print("i j ", i, j ,"chunk size is",
      #              chunk_grid[i][j].stringify(0,mip=mip))
      if overlap == 0:
          chunks.append(BoundingBox(chunk_grid[0][0].x_range(mip=mip)[0]-pad,
                                    chunk_grid[-1][-1].x_range(mip=mip)[1]+pad,
                                    chunk_grid[0][0].y_range(mip=mip)[0]-pad,
                                    chunk_grid[-1][-1].y_range(mip=mip)[1]+pad,
                                    mip=mip))
          return chunks
      start =0
      while(start+rows<len(chunk_grid)):
          print("start + row is", start+rows)
          chunks.append(BoundingBox(chunk_grid[start][0].x_range(mip=mip)[0]-pad,
                                    chunk_grid[start+rows-1][-1].x_range(mip=mip)[1]+pad,
                                    chunk_grid[start][0].y_range(mip=mip)[0]-pad,
                                    chunk_grid[start+rows-1][-1].y_range(mip=mip)[1]+pad,
                                    mip=mip))
          start = start + rows - overlap
      if start<len(chunk_grid):
          chunks.append(BoundingBox(chunk_grid[start][0].x_range(mip=mip)[0]-pad,
                                    chunk_grid[-1][-1].x_range(mip=mip)[1]+pad,
                                    chunk_grid[start][0].y_range(mip=mip)[0]-pad,
                                    chunk_grid[-1][-1].y_range(mip=mip)[1]+pad,
                                    mip=mip))
      return chunks

  def load_range_image(self, src_cv, dst_cv, z_range, bbox, mip, step, mask_cv=None,
                       mask_mip=0, mask_val=0):
      #print("---------------------------->> load image")
      batch = []
      for i in z_range:
          batch.append(tasks.LoadImageTask(src_cv, dst_cv, i, bbox, mip, step, mask_cv,
                                          mask_mip, mask_val))
      return batch

  def load_store_range_image(self, src_cv, dst_cv, z_range, bbox, mip, step,
                             pad, final_chunk, compress=None, mask_cv=None, mask_mip=0,
                             mask_val=0):
      batch = []
      for i in z_range:
          batch.append(tasks.LoadStoreImageTask(src_cv, dst_cv, i, bbox, mip, step, mask_cv,
                                          mask_mip, mask_val, pad, final_chunk,
                                               compress))
      return batch

  def store_random_image(self, src_cv, dst_cv, z_range, bbox, mip, step,
                             pad, final_chunk, compress=None, mask_cv=None, mask_mip=0,
                             mask_val=0):
      batch = []
      for i in z_range:
          batch.append(tasks.RandomStoreImageTask(dst_cv, i, mip, step, mask_cv,
                                          mask_mip, mask_val, pad, final_chunk,
                                               compress))
      return batch



  def load_part_image(self, image_cv, z, bbox, image_mip, mask_cv=None,
                       mask_mip=0, mask_val=0, to_tensor=True, affine=None):
      #tmp_device = self.device
      #self.device = 'cpu'
      if affine is not None:
          aff = affine[z]
          x_range = bbox.x_range(mip=0)
          y_range = bbox.y_range(mip=0)
          bbox =BoundingBox(x_range[0]-aff[:,2][1],
                            x_range[1]-aff[:,2][1],
                            y_range[0]-aff[:,2][0],
                            y_range[1]-aff[:,2][0], mip=0)
      image = self.get_image(image_cv, z, bbox, image_mip,
                             to_tensor=to_tensor, normalizer=None,
                             to_float=False, data_device='cpu')
      if mask_cv is not None:
        mask = self.get_mask(mask_cv, image_z, bbox,
                             src_mip=mask_mip,
                             dst_mip=image_mip, valid_val=mask_val)
        image = image.masked_fill_(mask, 0)
      #self.device = tmp_device
      return image

  def new_cloudsample_image(self, image, field):
      if is_identity(field):
        return image
      else:
        #image = grid_sample(image, field, padding_mode='zeros')
        model = nn.DataParallel(self.warp_model)
        image = model(image, field, 'zeros')
        return image

  def new_vector_vote(self, model_path, src_img, image_list, chunk_size, pad,
                      vvote_way, mip, inverse=False, serial=True,
                      head_crop=True, end_crop=True,
                      softmin_temp=None, blur_sigma=None):
    img_shape = src_img.shape
    x_len = img_shape[-2]
    y_len = img_shape[-1]
    padded_len = chunk_size + 2*pad
    x_chunk_number = (x_len - 2*pad) // chunk_size
    y_chunk_number = (y_len - 2*pad) // chunk_size
    #if(first_chunk):
    #    adjust = pad
    #else:
    #    adjust = 0
    image = torch.FloatTensor(1, 1, x_len-2*pad, y_len).zero_()
    dst_field = torch.FloatTensor(1,x_len, y_len, 2).zero_()
    #print("===============x_chunk_number is ", x_chunk_number)
    #elif head_crop == False and end_crop:
    #    offset = 0
    def coor(x,y):
        for i in range(x):
            for j in range(y):
                yield i ,j
        yield -1,-1
    #torch.cuda.synchronize()
    #start = time()
    gpu_num = torch.cuda.device_count()
    get_corr = coor(x_chunk_number, y_chunk_number)
    has_next = True
    while(has_next):
        coor_list = []
        xs, ys = next(get_corr)
        if(xs ==-1):
            break
        src_patch = src_img[...,xs*chunk_size:xs*chunk_size+padded_len,
                                ys*chunk_size:ys*chunk_size+padded_len]
        coor_list.append([xs, ys])
        for i in range(1, gpu_num):
            xs, ys = next(get_corr)
            if(xs == -1):
                has_next = False
                break
            else:
                tmp = src_img[...,xs*chunk_size:xs*chunk_size+padded_len,
                                ys*chunk_size:ys*chunk_size+padded_len]
                src_patch = torch.cat((src_patch, tmp),0)
                coor_list.append([xs, ys])
        src_patch = src_patch.cuda()
        src_patch = self.convert_to_float(src_patch)
        for i in range(vvote_way):
            if head_crop and end_crop:
                offset = (image_list[i].shape[-2] - src_img.shape[-2])//2
            elif head_crop and end_crop==False:
                offset = (image_list[i].shape[-2] - src_img.shape[-2])
            else:
                offset = 0;
            for j in range(len(coor_list)):
                xs, ys = coor_list[j]
                if j == 0:
                    tgt_patch = image_list[i][...,
                                offset+xs*chunk_size:offset+xs*chunk_size+padded_len,
                                              ys*chunk_size:ys*chunk_size+padded_len]
                else:
                    tmp = image_list[i][...,
                                offset+xs*chunk_size:offset+xs*chunk_size+padded_len,
                                              ys*chunk_size:ys*chunk_size+padded_len]
                    tgt_patch = torch.cat((tgt_patch, tmp),0)
            tgt_patch = tgt_patch.cuda()
            if tgt_patch.dtype == torch.uint8:
                tgt_patch = self.convert_to_float(tgt_patch)
            field = self.new_compute_field_chunk(model_path, src_patch,
                                                   tgt_patch)
            field = field[:,pad:-pad,pad:-pad,:]
        #    print("XXXXXXXXXXXXXXXx -----> field shape ", field.shape)
            if i==0:
                vv_fields = field.unsqueeze(0)
            else:
                vv_fields = torch.cat((vv_fields, field.unsqueeze(0)),0)
        #print("XXXXXXXXXXXXXXXx -----> vv_fields  shape", vv_fields.shape)
        vv_fields = vv_fields.permute(1,0,2,3,4)
        new_field = self.new_vector_vote_chunk(vv_fields, mip,
                                           softmin_temp=softmin_temp,
                                           blur_sigma=blur_sigma)
        new_field = new_field.to(device='cpu')
        for i in range(len(coor_list)):
            xs, ys = coor_list[i]
            dst_field[:,pad+xs*chunk_size:pad+xs*chunk_size+chunk_size,
                      pad+ys*chunk_size:pad+ys*chunk_size+chunk_size,:] = new_field[i:i+1,...]

    #torch.cuda.synchronize()
    #vv_end = time()
    #print("******* compute field and vv time is ", vv_end-start)
    print("finish compute  field and VV")
    #warp the image
    has_next = True
    get_corr = coor(x_chunk_number, y_chunk_number)
    image = self.warp_slice(chunk_size, pad, src_img, dst_field, get_corr)

    #while(has_next):
    #    coor_list = []
    #    xs, ys = next(get_corr)
    #    if(xs ==-1):
    #        break
    #    src_patch = src_img[...,xs*chunk_size:xs*chunk_size+padded_len,
    #                            ys*chunk_size:ys*chunk_size+padded_len]
    #    field = dst_field[:,xs*chunk_size:xs*chunk_size+padded_len,
    #                      ys*chunk_size:ys*chunk_size+padded_len,:]
    #    coor_list.append([xs, ys])
    #    for i in range(1, gpu_num):
    #        xs, ys = next(get_corr)
    #        if(xs == -1):
    #            has_next = False
    #            break
    #        else:
    #            tmp_s = src_img[...,xs*chunk_size:xs*chunk_size+padded_len,
    #                            ys*chunk_size:ys*chunk_size+padded_len]
    #            src_patch = torch.cat((src_patch, tmp_s),0)
    #            tmp_f = dst_field[:,xs*chunk_size:xs*chunk_size+padded_len,
    #                      ys*chunk_size:ys*chunk_size+padded_len,:]
    #            field = torch.cat((field, tmp_f),0)
    #            coor_list.append([xs, ys])
    #    src_patch = src_patch.to(device=self.device)
    #    src_patch = self.convert_to_float(src_patch)
    #    field = field.to(device=self.device)
    #    image_patch = self.new_cloudsample_image(src_patch, field)
    #    image_patch = image_patch[:,:,pad:-pad,pad:-pad]
    #    image_patch = image_patch.to(device='cpu')
    #    for i in range(len(coor_list)):
    #        xs, ys = coor_list[i]
    #        image[..., xs*chunk_size:xs*chunk_size+chunk_size,
    #              pad+ys*chunk_size:pad+ys*chunk_size+chunk_size]=image_patch[i:i+1,...]

    #torch.cuda.synchronize()
    #warp_end = time()
    #print("******** warp time is ", warp_end-vv_end)
    return image, dst_field


  #  #vector voting
  #  #for ys in range(0, y_chunk_number-1, 2):
  #  for ys in range(0, y_chunk_number, gpu_num):
  #      vv_fields = []
  #      src_patch = src_img[...,:padded_len, ys*chunk_size:ys*chunk_size+padded_len]
  #      if ys + gpu_num < y_chunk_number:
  #          for i in range(1, gpu_num):
  #              idx = ys + i
  #              tmp = src_img[...,:padded_len, idx*chunk_size:idx*chunk_size+padded_len]
  #              src_patch = torch.cat((src_patch, tmp),0)
  #      else:
  #          for idx in range(ys+1, y_chunk_number):
  #              tmp = src_img[...,:padded_len, idx*chunk_size:idx*chunk_size+padded_len]
  #              src_patch = torch.cat((src_patch, tmp),0)
  #      start_cf = time()
  #      #src_patch = src_patch.to(device=self.device)
  #      src_patch = src_patch.cuda()
  #      src_patch = self.convert_to_float(src_patch)
  #      for i in range(vvote_way):
  #          if head_crop and end_crop:
  #              offset = (image_list[i].shape[-2] - src_img.shape[-2])//2
  #          elif head_crop and end_crop==False:
  #              offset = (image_list[i].shape[-2] - src_img.shape[-2])
  #          else:
  #              offset = 0;
  #          #print("---------- in vv tgt image shape", image_list[i].shape)
  #          tgt_patch = image_list[i][..., offset:offset+padded_len,
  #                                    ys*chunk_size:ys*chunk_size+padded_len]
  #          if ys + gpu_num < y_chunk_number:
  #              for i in range(1, gpu_num):
  #                  idx = ys + i
  #                  tmp = image_list[i][..., offset:offset+padded_len,
  #                                    idx*chunk_size:idx*chunk_size+padded_len]
  #                  tgt_patch = torch.cat((tgt_patch, tmp),0)
  #          else:
  #              for idx in range(ys+1, y_chunk_number):
  #                  tmp = image_list[i][..., offset:offset+padded_len,
  #                                    idx*chunk_size:idx*chunk_size+padded_len]
  #                  tgt_patch = torch.cat((tgt_patch, tmp),0)
  #          #print("--------------in vv tgt patch shape", tgt_patch.shape)
  #          #torch.cuda.synchronize()
  #          #copy_s = time()
  #          #torch.cuda.synchronize()
  #          #copy_e = time()
  #          tgt_patch = tgt_patch.cuda()
  #          if tgt_patch.dtype == torch.uint8:
  #              tgt_patch = self.convert_to_float(tgt_patch)
  #          #print(" in vvoting src_patch size ", src_patch.shape, " tgt_patch",
  #          #     tgt_patch.shape)
  #          #print("---------- in vv tgt image shape", tgt_patch.shape)
  #          #torch.cuda.synchronize()
  #          #f_t = time()
  #          field = self.new_compute_field_chunk(model_path, src_patch,
  #                                                 tgt_patch)
  #          print("XXXXXXXXXXXXXXXx -----> field shape ", field.shape)
  #          #delete this line: field = field[:,pad:-pad,pad:-pad,:]
  #          #torch.cuda.synchronize()
  #          #f_t_e = time()
  #          #print(" each cf time", f_t_e - f_t)
  #          if i==0:
  #              vv_fields = field.unsqueeze(0)
  #          else:
  #              vv_fields = torch.cat((vv_fields, field.unsqueeze(0)),0)
  #          #if i == 0:
  #          #    for i in range(field.shape[0]):
  #          #        tmp = []
  #          #        tmp.append(field[i:i+1,...])
  #          #        vv_fields.append(tmp)
  #          #else:
  #          #    for i in range(field.shape[0]):
  #          #        vv_fields[i].append(field[i:i+1,...])

  #      #torch.cuda.synchronize()
  #      #vv_s = time()
  #      print("XXXXXXXXXXXXXXXx -----> vv_fields  shape", vv_fields.shape)
  #      vv_fields = vv_fields.permute(1,0,2,3,4)

  #      #for i in range(len(vv_fields)):
  #      #    vv_field = vv_fields[i]
  #      #    #print(" field shape", len(vv_field), vv_field[0].shape, vv_field[0])
  #      #    for j in range(len(vv_field)):
  #      #        dev = "cuda:"+str(i)
  #      #        #print("device is ", dev)
  #      #        vv_field[j] = vv_field[j].to(device=dev)
  #      #    print("start time", time(), "deivce is", vv_field[0].device)
  #      #    new_field = self.new_vector_vote_chunk(vv_field, mip,
  #      #                                   softmin_temp=softmin_temp,
  #      #                                   blur_sigma=blur_sigma)

  #      new_field = self.new_vector_vote_chunk(vv_fields, mip,
  #                                         softmin_temp=softmin_temp,
  #                                         blur_sigma=blur_sigma)
  #      print("xxxxxxxxxxxxxxxxxx  vv done")
  #      #torch.cuda.synchronize()
  #      #end_cf = time()
  #      #print("+++++++++++vv time", end_cf - vv_s)
  #      #print("=================compute", vvote_way, " chunk fields time",
  #      #      end_cf-start_cf)
  #      new_field = new_field[:, 0:-pad, pad:-pad,:]
  #      torch.cuda.synchronize()
  #      copy_s = time()
  #      new_field = new_field.to(device='cpu')
  #      torch.cuda.synchronize()
  #      copy_e = time()
  #      print("---------------------- vector field from GPU to CPU",
  #            copy_e-copy_s)
  #      for i in range(new_field.shape[0]):
  #          idx = ys+i
  #          dst_field[:,0:chunk_size+pad,
  #                pad+idx*chunk_size:pad+idx*chunk_size+chunk_size, :] = new_field[i:i+1,...]
  #      #dst_field[:,0:chunk_size+pad,
  #      #          pad+ys*chunk_size:pad+ys*chunk_size+chunk_size, :] = new_field

  #  #end = time()
  #  #print("================= first row in vv time is", end -start,
  #  #      "y chunk_number is", y_chunk_number, " x_chunk_number",
  #  #      x_chunk_number)
  #  print("================ first row finish")
  #  for xs in range(1, x_chunk_number-1, 3):
  #      for ys in range(y_chunk_number):
  #          vv_fields = []
  #          src_patch = src_img[...,xs*chunk_size:xs*chunk_size+padded_len,
  #                                  ys*chunk_size:ys*chunk_size+padded_len]
  #          src_patch1 = src_img[...,newx1*chunk_size:newx1*chunk_size+padded_len,
  #                                  ys*chunk_size:ys*chunk_size+padded_len]
  #          src_patch2 = src_img[...,newx2*chunk_size:newx2*chunk_size+padded_len,
  #                                  ys*chunk_size:ys*chunk_size+padded_len]
  #          src_patch = torch.cat((src_patch, src_patch1, src_patch2),0)
  #          src_patch = src_patch.to(device=self.device)
  #          src_patch = self.convert_to_float(src_patch)
  #          for i in range(vvote_way):
  #              if head_crop and end_crop:
  #                  offset = (image_list[i].shape[-2] - src_img.shape[-2])//2
  #              elif head_crop and end_crop==False:
  #                  offset = (image_list[i].shape[-2] - src_img.shape[-2])
  #              else:
  #                  offset = 0;
  #              tgt_patch = image_list[i][...,
  #                                        offset+xs*chunk_size:offset+xs*chunk_size+padded_len,
  #                                        ys*chunk_size:ys*chunk_size+padded_len]
  #              tgt_patch1 = image_list[i][...,
  #                                        offset+newx1*chunk_size:offset+newx1*chunk_size+padded_len,
  #                                        ys*chunk_size:ys*chunk_size+padded_len]
  #              tgt_patch2 = image_list[i][...,
  #                                        offset+newx2*chunk_size:offset+newx2*chunk_size+padded_len,
  #                                        ys*chunk_size:ys*chunk_size+padded_len]
  #              tgt_patch = torch.cat((tgt_patch, tgt_patch1, tgt_patch2), 0)
  #              tgt_patch = tgt_patch.to(device=self.device)
  #              if tgt_patch.dtype == torch.uint8:
  #                  tgt_patch = self.convert_to_float(tgt_patch)
  #              field = self.new_compute_field_chunk(model_path, src_patch,
  #                                                     tgt_patch)
  #              field = field[:,pad:-pad,pad:-pad,:]
  #              vv_fields.append(field)
  #  #        new_field = self.new_vector_vote_chunk(vv_fields, mip,
  #  #                                           softmin_temp=softmin_temp,
  #  #                                           blur_sigma=blur_sigma)
  #  #        new_field = new_field.to(device='cpu')
  #  #        dst_field[:,pad+xs*chunk_size:pad+xs*chunk_size+chunk_size,
  #  #                  pad+ys*chunk_size:pad+ys*chunk_size+chunk_size,:] = new_field
  #  #new_end = time()
  #  #print("================= middle row in vv time is", new_end - end)

  #  for xs in [x_chunk_number-1]:
  #      for ys in range(y_chunk_number):
  #          vv_fields = []
  #          src_patch = src_img[...,xs*chunk_size:xs*chunk_size+padded_len,
  #                                  ys*chunk_size:ys*chunk_size+padded_len]
  #          src_patch = src_patch.to(device=self.device)
  #          src_patch = self.convert_to_float(src_patch)
  #          for i in range(vvote_way):
  #              if head_crop and end_crop:
  #                  offset = (image_list[i].shape[-2] - src_img.shape[-2])//2
  #              elif head_crop and end_crop==False:
  #                  offset = (image_list[i].shape[-2] - src_img.shape[-2])
  #              else:
  #                  offset = 0;

  #              tgt_patch = image_list[i][...,
  #                                        offset+xs*chunk_size:offset+xs*chunk_size+padded_len,
  #                                        ys*chunk_size:ys*chunk_size+padded_len]
  #              tgt_patch = tgt_patch.to(device=self.device)
  #              if tgt_patch.dtype == torch.uint8:
  #                  tgt_patch = self.convert_to_float(tgt_patch)
  #              field = self.new_compute_field_chunk(model_path, src_patch,
  #                                                     tgt_patch)
  #              #field = field[:,pad:-pad,pad:-pad,:]
  #              vv_fields.append(field)
  #          new_field = self.new_vector_vote_chunk(vv_fields, mip,
  #                                             softmin_temp=softmin_temp,
  #                                             blur_sigma=blur_sigma)
  #          # to verify: wherther non-square size is good
  #          new_field = new_field[:,pad:,pad:-pad,:]
  #          new_field = new_field.to(device='cpu')
  #          #print("--------field shape", field.shape )
  #          dst_field[:,pad+xs*chunk_size:pad+xs*chunk_size+chunk_size+pad,
  #                    pad+ys*chunk_size:pad+ys*chunk_size+chunk_size,:] = new_field
  #  torch.cuda.synchronize()
  #  last_end = time()
  #  print("================= whole vv time is", last_end - start)
  #  #print("================= last row in vv time is", last_end - new_end)
  # 
  #   
  #  #warp the image
  #  for xs in range(x_chunk_number):
  #      for ys in range(y_chunk_number):
  #          src_patch = src_img[...,xs*chunk_size:xs*chunk_size+padded_len,
  #                              ys*chunk_size:ys*chunk_size+padded_len]
  #          field = dst_field[:,xs*chunk_size:xs*chunk_size+padded_len,
  #                            ys*chunk_size:ys*chunk_size+padded_len,:]
  #          src_patch = src_patch.to(device=self.device)
  #          src_patch = self.convert_to_float(src_patch)
  #          field = field.to(device=self.device)
  #          #field = self.invert_field(field)
  #          image_patch = self.new_cloudsample_image(src_patch, field)
  #          image_patch = image_patch[:,:,pad:-pad,pad:-pad]
  #          #image_patch = self.convert_to_uint8(image_patch)
  #          image_patch = image_patch.to(device='cpu')
  #          image[..., xs*chunk_size:xs*chunk_size+chunk_size,
  #                pad+ys*chunk_size:pad+ys*chunk_size+chunk_size] = image_patch
  #  torch.cuda.synchronize()
  #  warp_end = time()
  #  print("====================== warp end time", warp_end - last_end)
  #  return image, dst_field

  def new_vector_vote_chunk(self, fields, mip,softmin_temp=None, blur_sigma=None):
    if not softmin_temp:
      w = 0.99
      d = 2**mip
      n = len(fields)
      m = int(binom(n, (n+1)//2)) - 1
      softmin_temp = 2**mip
    #print("fields shape", fields.shape)
    model = nn.DataParallel(self.vvmodel)
    #res_field = model(field_tensor=fields, softmin_temp=softmin_temp, blur_sigma=blur_sigma)
    res_field = model(fields, softmin_temp, blur_sigma)
    #print("--------------------final field shape", res_field.shape)
    return res_field
    #return vector_vote(fields, softmin_temp=softmin_temp, blur_sigma=blur_sigma)



  ##########################
  # Chunking & BoundingBox #
  ##########################

  def break_into_chunks_grid(self, bbox, chunk_size, offset, mip, max_mip=12):
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
    if chunk_size[0] > self.chunk_size[0] or chunk_size[1] > self.chunk_size[1]:
      chunk_size = self.chunk_size 

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
      x_chunks = [] 
      for ys in range(calign_y_range[0], calign_y_range[1], chunk_size[1]):
        x_chunks.append(BoundingBox(xs, xs + chunk_size[0],
                                 ys, ys + chunk_size[1],
                                 mip=mip, max_mip=max_mip))
      chunks.append(x_chunks)
    return chunks

  def break_into_chunks(self, bbox, chunk_size, offset, mip, max_mip=12):
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
    if chunk_size[0] > self.chunk_size[0] or chunk_size[1] > self.chunk_size[1]:
      chunk_size = self.chunk_size 

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

  def adjust_bbox(self, bbox, dis):
      padded_bbox = deepcopy(bbox)
      x_range = padded_bbox.x_range(mip=0)
      y_range = padded_bbox.y_range(mip=0)
      new_bbox = BoundingBox(x_range[0] + dis[0], x_range[1] + dis[0],
                             y_range[0] + dis[1], y_range[1] + dis[1],
                             mip=0)
      return new_bbox

  ##############
  # IO methods #
  ##############

  def get_model_archive(self, model_path):
    """Load a model stored in the repo with its relative path

    TODO: evict old models from self.models

    Args:
       model_path: str for relative path to model directory

    Returns:
       the ModelArchive at that model_path
    """
    if model_path in self.model_archives:
      #print('Loading model {0} from cache'.format(model_path), flush=True)
      return self.model_archives[model_path]
    else:
      print('Adding model {0} to the cache'.format(model_path), flush=True)
      path = Path(model_path)
      model_name = path.stem
      archive = ModelArchive(model_name)
      self.model_archives[model_path] = archive
      return archive

  #######################
  # Image IO + handlers #
  #######################

  def get_mask(self, cv, z, bbox, src_mip, dst_mip, valid_val, to_tensor=True):
    start = time()
    data = self.get_data(cv, z, bbox, src_mip=src_mip, dst_mip=dst_mip, 
                             to_float=False, to_tensor=to_tensor, normalizer=None)
    mask = data == valid_val
    end = time()
    diff = end - start
    print('get_mask: {:.3f}'.format(diff), flush=True) 
    return mask

  def get_image(self, cv, z, bbox, mip, to_tensor=True, normalizer=None,
                to_float=True, data_device=None):
    print('get_image for {0}'.format(bbox.stringify(z)), flush=True)
    start = time()
    image = self.get_data(cv, z, bbox, src_mip=mip, dst_mip=mip,
                          to_float=to_float, to_tensor=to_tensor,
                          normalizer=normalizer, data_device=data_device)
    end = time()
    diff = end - start
    print('get_image: {:.3f}'.format(diff), flush=True) 
    return image

  def get_masked_image(self, image_cv, z, bbox, image_mip, mask_cv, mask_mip, mask_val,
                             to_tensor=True, normalizer=None):
    """Get image with mask applied
    """
    start = time()
    image = self.get_image(image_cv, z, bbox, image_mip,
                           to_tensor=True, normalizer=normalizer)
    if mask_cv is not None:
      mask = self.get_mask(mask_cv, z, bbox, 
                           src_mip=mask_mip,
                           dst_mip=image_mip, valid_val=mask_val)
      image = image.masked_fill_(mask, 0)
    if not to_tensor:
      image = image.cpu().numpy()
    end = time()
    diff = end - start
    print('get_masked_image: {:.3f}'.format(diff), flush=True) 
    return image

  def get_composite_image(self, image_cv, z_list, bbox, image_mip,
                                mask_cv, mask_mip, mask_val,
                                to_tensor=True, normalizer=None):
    """Collapse a stack of 2D image into a single 2D image, by consecutively
        replacing black pixels (0) in the image of the first z_list entry with
        non-black pixels from of the consecutive z_list entries images.

    Args:
       image_cv: MiplessCloudVolume where images are stored
       z_list: list of image indices processed in the given order
       bbox: BoundingBox defining data range
       image_mip: int MIP level of the image data to process
       mask_cv: MiplessCloudVolume where masks are stored, or None if no mask
        should be used
       mask_mip: int MIP level of the mask, ignored if ``mask_cv`` is None
       mask_val: The mask value that specifies regions to be blackened, ignored
        if ``mask_cv`` is None.
       to_tensor: output will be torch.tensor
       #TODO normalizer: callable function to adjust the contrast of each image
    """

    # Retrieve image stack
    assert len(z_list) > 0

    combined = self.get_masked_image(image_cv, z_list[0], bbox, image_mip,
                                     mask_cv, mask_mip, mask_val,
                                     to_tensor=to_tensor, normalizer=normalizer)
    for z in z_list[1:]:
      tmp = self.get_masked_image(image_cv, z, bbox, image_mip,
                                  mask_cv, mask_mip, mask_val,
                                  to_tensor=to_tensor, normalizer=normalizer)
      black_mask = combined == 0
      combined[black_mask] = tmp[black_mask]

    return combined

  def get_data(self, cv, z, bbox, src_mip, dst_mip, to_float=True, 
                     to_tensor=True, normalizer=None, data_device=None):
    """Retrieve CloudVolume data. Returns 4D ndarray or tensor, BxCxWxH
    
    Args:
       cv_key: string to lookup CloudVolume
       bbox: BoundingBox defining data range
       src_mip: mip of the CloudVolume data
       dst_mip: mip of the output mask (dictates whether to up/downsample)
       to_float: output should be float32
       to_tensor: output will be torch.tensor
       normalizer: callable function to adjust the contrast of the image

    Returns:
       image from CloudVolume in region bbox at dst_mip, with contrast adjusted,
       if normalizer is specified, and as a uint8 or float32 torch tensor or numpy, 
       as specified
    """
    if data_device == None:
        data_device = self.device
    x_range = bbox.x_range(mip=src_mip)
    y_range = bbox.y_range(mip=src_mip)
    #cv.green_threads = True
    data = cv[src_mip][x_range[0]:x_range[1], y_range[0]:y_range[1], z]
    data = np.transpose(data, (2,3,0,1))
    if to_float:
      data = np.divide(data, float(255.0), dtype=np.float32)
    if (normalizer is not None) and (not is_blank(data)):
      print('Normalizing image')
      start = time()
      data = torch.from_numpy(data)
      data = data.to(device=data_device)
      data = normalizer(data).reshape(data.shape)
      end = time()
      diff = end - start
      print('normalizer: {:.3f}'.format(diff), flush=True) 
    # convert to tensor if requested, or if up/downsampling required
    if to_tensor | (src_mip != dst_mip):
      if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
      data = data.to(device=data_device)
      if src_mip != dst_mip:
        # k = 2**(src_mip - dst_mip)
        size = (bbox.y_size(dst_mip), bbox.x_size(dst_mip))
        if not isinstance(data, torch.cuda.ByteTensor): #TODO: handle device
          data = interpolate(data, size=size, mode='bilinear')
        else:
          data = data.type('torch.cuda.DoubleTensor')
          data = interpolate(data, size=size, mode='nearest')
          data = data.type('torch.cuda.ByteTensor')
      if not to_tensor:
        data = data.cpu().numpy()
    
    return data
  
  def get_data_range(self, cv, z_range, bbox, src_mip, dst_mip, to_tensor=True):
    """Retrieve CloudVolume data. Returns 4D tensor, BxCxWxH
    
    Args:
       cv_key: string to lookup CloudVolume
       bbox: BoundingBox defining data range
       src_mip: mip of the CloudVolume data
       dst_mip: mip of the output mask (dictates whether to up/downsample)
       to_tensor: output will be torch.tensor
       #TODO normalizer: callable function to adjust the contrast of the image
    """
    x_range = bbox.x_range(mip=src_mip)
    y_range = bbox.y_range(mip=src_mip)
    data = cv[src_mip][x_range[0]:x_range[1], y_range[0]:y_range[1], z_range]
    data = np.transpose(data, (2,3,0,1))
    if isinstance(data, np.ndarray):
      data = torch.from_numpy(data)
    data = data.to(device=self.device)
    if src_mip != dst_mip:
      # k = 2**(src_mip - dst_mip)
      size = (bbox.y_size(dst_mip), bbox.x_size(dst_mip))
      if not isinstance(data, torch.cuda.ByteTensor): #TODO: handle device
        data = interpolate(data, size=size, mode='bilinear')
      else:
        data = data.type('torch.cuda.DoubleTensor')
        data = interpolate(data, size=size, mode='nearest')
        data = data.type('torch.cuda.ByteTensor')
    if not to_tensor:
      data = data.cpu().numpy()
    
    return data

  def save_image(self, float_patch, cv, z, bbox, mip, to_uint8=True):
    x_range = bbox.x_range(mip=mip)
    y_range = bbox.y_range(mip=mip)
    patch = np.transpose(float_patch, (2,3,0,1))
    print("SAVE IMG z is", z, "save image patch at mip", mip, "range at mip0", bbox.stringify(z))
    if to_uint8:
      patch = (np.multiply(patch, 255)).astype(np.uint8)
    cv[mip][x_range[0]:x_range[1], y_range[0]:y_range[1], z] = patch

  def save_image_batch(self, cv, z_range, float_patch, bbox, mip, to_uint8=True):
    x_range = bbox.x_range(mip=mip)
    y_range = bbox.y_range(mip=mip)
    print("type of float_patch", type(float_patch), "shape", float_patch.shape)
    patch = np.transpose(float_patch, (2,3,0,1))
    # patch = np.transpose(float_patch, (2,1,0))[..., np.newaxis]
    if to_uint8:
        patch = (np.multiply(patch, 255)).astype(np.uint8)
    print("patch shape", patch.shape)
    cv[mip][x_range[0]:x_range[1], y_range[0]:y_range[1],
            z_range[0]:z_range[1]] = patch

  #######################
  # Field IO + handlers #
  #######################
  def get_field(self, cv, z, bbox, mip, relative=False, to_tensor=True, as_int16=True):
    """Retrieve vector field from CloudVolume.

    Args
      CV: MiplessCloudVolume storing vector field as MIP0 residuals in X,Y,Z,2 order
      Z: int for section index
      BBOX: BoundingBox for X & Y extent of the field to retrieve
      MIP: int for resolution at which to pull the vector field
      RELATIVE: bool indicating whether to convert MIP0 residuals to relative residuals
        from [-1,1] based on residual location within shape of the BBOX
      TO_TENSOR: bool indicating whether to return FIELD as a torch tensor

    Returns
      FIELD: vector field with dimensions of BBOX at MIP, with RELATIVE residuals &
        as TO_TENSOR, using convention (Z,Y,X,2) 

    Note that the grid convention for torch.grid_sample is (N,H,W,2), where the
    components in the final dimension are (x,y). We are NOT altering it here.
    """
    x_range = bbox.x_range(mip=mip)
    y_range = bbox.y_range(mip=mip)
    print('get_field from {bbox}, z={z}, MIP{mip} to {path}'.format(bbox=bbox,
                                 z=z, mip=mip, path=cv.path))
    field = cv[mip][x_range[0]:x_range[1], y_range[0]:y_range[1], z]
    field = np.transpose(field, (2,0,1,3))
    if as_int16:
      field = np.float32(field) / 4
    if relative:
      field = self.abs_to_rel_residual(field, bbox, mip)
    if to_tensor:
      field = torch.from_numpy(field)
      return field.to(device=self.device)
    else:
      return field 

  def save_field(self, field, cv, z, bbox, mip, relative, as_int16=True):
    """Save vector field to CloudVolume.

    Args
      field: ndarray vector field with dimensions of bbox at mip with absolute MIP0 
        residuals, using grid_sample convention of (Z,Y,X,2), where the components in 
        the final dimension are (x,y).
      cv: MiplessCloudVolume to store vector field as MIP0 residuals in X,Y,Z,2 order
      z: int for section index
      bbox: BoundingBox for X & Y extent of the field to be stored
      mip: int for resolution at which to store the vector field
      relative: bool indicating whether to convert MIP0 residuals to relative residuals
        from [-1,1] based on residual location within shape of the bbox 
      as_int16: bool indicating whether vectors should be saved as int16
    """
    if relative: 
      field = field * (field.shape[-2] / 2) * (2**mip)
    # field = field.data.cpu().numpy() 
    x_range = bbox.x_range(mip=mip)
    y_range = bbox.y_range(mip=mip)
    field = np.transpose(field, (1,2,0,3))
    print('save_field for {0} at MIP{1} to {2}'.format(bbox.stringify(z),
                                                       mip, cv.path))
    if as_int16:
      if(np.max(field) > 8192 or np.min(field) < -8191):
        print('Value in field is out of range of int16 max: {}, min: {}'.format(
                                               np.max(field),np.min(field)), flush=True)
      field = np.int16(field * 4)
    #print("**********field shape is ", field.shape, type(field[0,0,0,0]))
    cv[mip][x_range[0]:x_range[1], y_range[0]:y_range[1], z] = field

  def rel_to_abs_residual(self, field, mip):    
    """Convert vector field from relative space [-1,1] to absolute MIP0 space
    """
    return field * (field.shape[-2] / 2) * (2**mip)

  def abs_to_rel_residual(self, field, bbox, mip):
    """Convert vector field from absolute MIP0 space to relative space [-1,1]
    """
    x_fraction = bbox.x_size(mip=0) * 0.5
    y_fraction = bbox.y_size(mip=0) * 0.5
    rel_residual = deepcopy(field)
    rel_residual[:, :, :, 0] /= x_fraction
    rel_residual[:, :, :, 1] /= y_fraction
    return rel_residual

  def avg_field(self, field):
    favg = field.sum() / (torch.nonzero(field).size(0) + self.eps)
    return favg

  def profile_field(self, field):
    avg_x = self.avg_field(field[0,...,0])
    avg_y = self.avg_field(field[0,...,1])
    return torch.Tensor([avg_x, avg_y])

  #############################
  # CloudVolume chunk methods #
  #############################

  def compute_field_chunk(self, model_path, src_cv, tgt_cv, src_z, tgt_z, bbox, mip, pad, 
                          src_mask_cv=None, src_mask_mip=0, src_mask_val=0,
                          tgt_mask_cv=None, tgt_mask_mip=0, tgt_mask_val=0,
                          tgt_alt_z=None, prev_field_cv=None, prev_field_z=None,
                          prev_field_inverse=False):
    """Run inference with SEAMLeSS model on two images stored as CloudVolume regions.

    Args:
      model_path: str for relative path to model directory
      src_z: int of section to be warped
      src_cv: MiplessCloudVolume with source image
      tgt_z: int of section to be warped to
      tgt_cv: MiplessCloudVolume with target image
      bbox: BoundingBox for region of both sections to process
      mip: int of MIP level to use for bbox
      pad: int for amount of padding to add to the bbox before processing
      mask_cv: MiplessCloudVolume with mask to be used for both src & tgt image
      prev_field_cv: if specified, a MiplessCloudVolume containing the
                     previously predicted field to be profile and displace
                     the src chunk

    Returns:
      field with MIP0 residuals with the shape of bbox at MIP mip (np.ndarray)
    """
    archive = self.get_model_archive(model_path)
    model = archive.model
    normalizer = archive.preprocessor
    print('compute_field for {0} to {1}'.format(bbox.stringify(src_z),
                                                bbox.stringify(tgt_z)))
    print('pad: {}'.format(pad))
    padded_bbox = deepcopy(bbox)
    padded_bbox.uncrop(pad, mip=mip)

    if prev_field_cv is not None:
        field = self.get_field(prev_field_cv, prev_field_z, padded_bbox, mip,
                           relative=False, to_tensor=True)
        if prev_field_inverse:
          field = -field
        distance = self.profile_field(field)
        print('Displacement adjustment: {} px'.format(distance))
        distance = (distance // (2 ** mip)) * 2 ** mip
        new_bbox = self.adjust_bbox(padded_bbox, distance.flip(0))
    else:
        distance = torch.Tensor([0, 0])
        new_bbox = padded_bbox

    tgt_z = [tgt_z]
    if tgt_alt_z is not None:
      try:
        tgt_z.extend(tgt_alt_z)
      except TypeError:
        tgt_z.append(tgt_alt_z)
      print('alternative target slices:', tgt_alt_z)

    src_patch = self.get_masked_image(src_cv, src_z, new_bbox, mip,
                                mask_cv=src_mask_cv, mask_mip=src_mask_mip,
                                mask_val=src_mask_val,
                                to_tensor=True, normalizer=normalizer)
    tgt_patch = self.get_composite_image(tgt_cv, tgt_z, padded_bbox, mip,
                                mask_cv=tgt_mask_cv, mask_mip=tgt_mask_mip,
                                mask_val=tgt_mask_val,
                                to_tensor=True, normalizer=normalizer)
    print('src_patch.shape {}'.format(src_patch.shape))
    print('tgt_patch.shape {}'.format(tgt_patch.shape))

    # Running the model is the only part that will increase memory consumption
    # significantly - only incrementing the GPU lock here should be sufficient.
    if self.gpu_lock is not None:
      self.gpu_lock.acquire()
      print("Process {} acquired GPU lock".format(os.getpid()))

    try:
      print("GPU memory allocated: {}, cached: {}".format(torch.cuda.memory_allocated(), torch.cuda.memory_cached()))

      # model produces field in relative coordinates
      field = model(src_patch, tgt_patch)
      print("GPU memory allocated: {}, cached: {}".format(torch.cuda.memory_allocated(), torch.cuda.memory_cached()))
      field = self.rel_to_abs_residual(field, mip)
      field = field[:,pad:-pad,pad:-pad,:]
      field += distance.to(device=self.device)
      field = field.data.cpu().numpy()
      # clear unused, cached memory so that other processes can allocate it
      torch.cuda.empty_cache()

      print("GPU memory allocated: {}, cached: {}".format(torch.cuda.memory_allocated(), torch.cuda.memory_cached()))
    finally:
      if self.gpu_lock is not None:
        print("Process {} releasing GPU lock".format(os.getpid()))
        self.gpu_lock.release()

    return field

  def predict_image(self, cm, model_path, src_cv, dst_cv, z, mip, bbox,
                    chunk_size, prefix=''):
    start = time()
    chunks = self.break_into_chunks(bbox, chunk_size,
                                    cm.dst_voxel_offsets[mip], mip=mip,
                                    max_mip=cm.num_scales)
    print("\nfold detect\n"
          "model {}\n"
          "src {}\n"
          "dst {}\n"
          "z={} \n"
          "MIP{}\n"
          "{} chunks\n".format(model_path, src_cv, dst_cv, z,
                               mip, len(chunks)), flush=True)
    if prefix == '':
      prefix = '{}'.format(mip)
    batch = []
    for patch_bbox in chunks:
      batch.append(tasks.PredictImgTask(model_path, src_cv, dst_cv, z, mip,
                                        patch_bbox, prefix))
    return batch

  def predict_image_chunk(self, model_path, src_cv, z, mip, bbox):
    archive = self.get_model_archive(model_path, readonly=2)
    model = archive.model
    image = self.get_image(src_cv, z, bbox, mip, to_tensor=True)
    new_image = model(image)
    return new_image


  def vector_vote_chunk(self, pairwise_cvs, vvote_cv, z, bbox, mip, 
                        inverse=False, serial=True, softmin_temp=None,
                        blur_sigma=None):
    """Compute consensus vector field using pairwise vector fields with earlier sections. 

    Vector voting requires that vector fields be composed to a common section
    before comparison: inverse=False means that the comparison will be based on 
    composed vector fields F_{z,compose_start}, while inverse=True will be
    F_{compose_start,z}.

    TODO:
       Reimplement field_cache

    Args:
       pairwise_cvs: dict of MiplessCloudVolumes, indexed by their z_offset
       vvote_cv: MiplessCloudVolume where vector-voted field will be stored 
       z: int for section index to be vector voted 
       bbox: BoundingBox for region where all fields will be loaded/written
       mip: int for MIP level of fields
       softmin_temp: softmin temperature (default will be 2**mip)
       inverse: bool indicating if pairwise fields are to be treated as inverse fields 
       serial: bool inreadonly=Truedicating to if a previously composed field is 
        not necessary
       softmin_temp: temperature to use for the softmin in vector voting; default None
        will use formula based on MIP level
       blur_sigma: std dev of Gaussian kernel by which to blur the vector vote inputs;
        default None means no blurring
       
    """
    fields = []
    for z_offset, f_cv in pairwise_cvs.items():
      if serial:
        F = self.get_field(f_cv, z, bbox, mip, relative=False, to_tensor=True)
      else:
        G_cv = vvote_cv
        if inverse:
          f_z = z+z_offset
          G_z = z+z_offset
          F = self.get_composed_field(f_cv, G_cv, f_z, G_z, bbox, mip, mip, mip)
        else:
          f_z = z
          G_z = z+z_offset
          F = self.get_composed_field(G_cv, f_cv, G_z, f_z, bbox, mip, mip, mip)
      fields.append(F)
    # assign weight w if the difference between majority vector similarities are d
    if not softmin_temp:
      w = 0.99
      d = 2**mip
      n = len(fields)
      m = int(binom(n, (n+1)//2)) - 1
      softmin_temp = 2**mip
    return vector_vote(fields, softmin_temp=softmin_temp, blur_sigma=blur_sigma)

  def invert_field(self, z, src_cv, dst_cv, bbox, mip, pad, model_path):
    """Compute the inverse vector field for a given bbox 

    Args:
       z: int for section index to be processed
       src_cv: MiplessCloudVolume where the field to be inverted is stored
       dst_cv: MiplessCloudVolume where the inverted field will be stored
       bbox: BoundingBox for region to be processed
       mip: int for MIP level to be processed
       pad: int for additional bbox padding to use during processing
       model_path: string for relative path to the inverter model; if blank, then use
        the runtime optimizer
    """
    padded_bbox = deepcopy(bbox)
    padded_bbox.uncrop(pad, mip=mip)
    f = self.get_field(src_cv, z, padded_bbox, mip,
                       relative=True, to_tensor=True, as_int16=as_int16)
    print('invert_field shape: {0}'.format(f.shape))
    start = time()
    if model_path:
      archive = self.get_model_archive(model_path)
      model = archive.model
      invf = model(f)
    else:
      # use optimizer if no model provided
      invf = invert(f)
    invf = self.rel_to_abs_residual(invf, mip=mip)
    invf = invf[:,pad:-pad, pad:-pad,:] 
    end = time()
    print (": {} sec".format(end - start))
    invf = invf.data.cpu().numpy() 
    self.save_field(dst_cv, z, invf, bbox, mip, relative=True, as_int16=as_int16) 

  def cloudsample_image(self, image_cv, field_cv, image_z, field_z,
                        bbox, image_mip, field_mip, mask_cv=None,
                        mask_mip=0, mask_val=0, affine=None,
                        use_cpu=False):
      """Wrapper for torch.nn.functional.gridsample for CloudVolume image objects

      Args:
        z: int for section index to warp
        image_cv: MiplessCloudVolume storing the image
        field_cv: MiplessCloudVolume storing the vector field
        bbox: BoundingBox for output region to be warped
        image_mip: int for MIP of the image
        field_mip: int for MIP of the vector field; must be >= image_mip.
         If field_mip > image_mip, the field will be upsampled.
        aff: 2x3 ndarray defining affine transform at MIP0 with which to precondition
         the field. If None, then will be ignored (treated as the identity).

      Returns:
        warped image with shape of bbox at MIP image_mip
      """
      if use_cpu:
          self.device = 'cpu'
      assert(field_mip >= image_mip)
      pad = 256
      padded_bbox = deepcopy(bbox)
      print('Padding by {} at MIP{}'.format(pad, image_mip))
      padded_bbox.uncrop(pad, mip=image_mip)

      # Load initial vector field
      field = self.get_field(field_cv, field_z, padded_bbox, field_mip,
                             relative=False, to_tensor=True)
      if field_mip > image_mip:
        field = upsample_field(field, field_mip, image_mip)

      if affine is not None:
        # PyTorch conventions are column, row order (y, then x) so flip
        # the affine matrix and offset
        affine = torch.Tensor(affine).to(field.device)
        affine = affine.flip(0)[:, [1, 0, 2]]  # flip x and y
        offset_y, offset_x = padded_bbox.get_offset(mip=0)

        ident = self.rel_to_abs_residual(
            identity_grid(field.shape, device=field.device), image_mip)

        field += ident
        field[..., 0] += offset_x
        field[..., 1] += offset_y
        field = torch.tensordot(
            affine[:, 0:2], field, dims=([1], [3])).permute(1, 2, 3, 0)
        field[..., :] += affine[:, 2]
        field[..., 0] -= offset_x
        field[..., 1] -= offset_y
        field -= ident

      if is_identity(field):
        image = self.get_image(image_cv, image_z, bbox, image_mip,
                               to_tensor=True, normalizer=None)
        if mask_cv is not None:
          mask = self.get_mask(mask_cv, image_z, bbox,
                               src_mip=mask_mip,
                               dst_mip=image_mip, valid_val=mask_val)
          image = image.masked_fill_(mask, 0)
        return image
      else:
        distance = self.profile_field(field)
        distance = (distance // (2 ** image_mip)) * 2 ** image_mip
        new_bbox = self.adjust_bbox(padded_bbox, distance.flip(0))

        field -= distance.to(device = self.device)
        field = self.abs_to_rel_residual(field, padded_bbox, image_mip)
        field = field.to(device = self.device)

        image = self.get_masked_image(image_cv, image_z, new_bbox, image_mip,
                                      mask_cv=mask_cv, mask_mip=mask_mip,
                                      mask_val=mask_val,
                                      to_tensor=True, normalizer=None)
        image = grid_sample(image, field, padding_mode='zeros')
        image = image[:,:,pad:-pad,pad:-pad]
        return image


  def cloudsample_compose(self, f_cv, g_cv, f_z, g_z, bbox, f_mip, g_mip,
                          dst_mip, factor=1., affine=None, pad=256):
      """Wrapper for torch.nn.functional.gridsample for CloudVolume field objects.

      Gridsampling a field is a composition, such that f(g(x)).

      Args:
         f_cv: MiplessCloudVolume storing the vector field to do the warping
         g_cv: MiplessCloudVolume storing the vector field to be warped
         bbox: BoundingBox for output region to be warped
         f_z, g_z: int for section index from which to read fields
         f_mip, g_mip: int for MIPs of the input fields
         dst_mip: int for MIP of the desired output field
         factor: float to multiply the f vector field by
         affine: an additional affine matrix to be composed before the fields
           If a is the affine matrix, then rendering the resulting field would
           be equivalent to
             f(g(a(x)))
         pad: number of pixels to pad at dst_mip

      Returns:
         composed field
      """
      assert(f_mip >= dst_mip)
      assert(g_mip >= dst_mip)
      padded_bbox = deepcopy(bbox)
      print('Padding by {} at MIP{}'.format(pad, dst_mip))
      padded_bbox.uncrop(pad, mip=dst_mip)
      # Load warper vector field
      f = self.get_field(f_cv, f_z, padded_bbox, f_mip,
                             relative=False, to_tensor=True)
      f = f * factor
      if f_mip > dst_mip:
        f = upsample_field(f, f_mip, dst_mip)

      if is_identity(f):
        g = self.get_field(g_cv, g_z, padded_bbox, g_mip,
                           relative=False, to_tensor=True)
        if g_mip > dst_mip:
            g = upsample_field(g, g_mip, dst_mip)
        return g

      distance = self.profile_field(f)
      distance = (distance // (2 ** g_mip)) * 2 ** g_mip
      new_bbox = self.adjust_bbox(padded_bbox, distance.flip(0))

      f -= distance.to(device = self.device)
      f = self.abs_to_rel_residual(f, padded_bbox, dst_mip)
      f = f.to(device = self.device)

      g = self.get_field(g_cv, g_z, new_bbox, g_mip,
                         relative=False, to_tensor=True)
      if g_mip > dst_mip:
        g = upsample_field(g, g_mip, dst_mip)
      g = self.abs_to_rel_residual(g, padded_bbox, dst_mip)
      h = compose_fields(f, g)
      h = self.rel_to_abs_residual(h, dst_mip)
      h += distance.to(device=self.device)
      h = h[:,pad:-pad,pad:-pad,:]

      if affine is not None:
        # PyTorch conventions are column, row order (y, then x) so flip
        # the affine matrix and offset
        affine = torch.Tensor(affine).to(f.device)
        affine = affine.flip(0)[:, [1, 0, 2]]  # flip x and y
        offset_y, offset_x = padded_bbox.get_offset(mip=0)

        ident = self.rel_to_abs_residual(
            identity_grid(f.shape, device=f.device), dst_mip)

        h += ident
        h[..., 0] += offset_x
        h[..., 1] += offset_y
        h = torch.tensordot(
            affine[:, 0:2], h, dims=([1], [3])).permute(1, 2, 3, 0)
        h[..., :] += affine[:, 2]
        h[..., 0] -= offset_x
        h[..., 1] -= offset_y
        h -= ident

      return h

  def cloudsample_multi_compose(self, field_list, z_list, bbox, mip_list,
                                dst_mip, factors=None, pad=256):
    """Compose a list of field CloudVolumes

    This takes a list of fields
    field_list = [f_0, f_1, ..., f_n]
    and composes them to get
    f_0 ⚬ f_1 ⚬ ... ⚬ f_n ~= f_0(f_1(...(f_n)))

    Args:
       field_list: list of MiplessCloudVolume storing the vector fields
       z_list: int or list of ints for section indices to read fields
       bbox: BoundingBox for output region to be warped
       mip_list: int or list of ints for MIPs of the input fields
       dst_mip: int for MIP of the desired output field
       pad: number of pixels to pad at dst_mip
       factors: floats to multiply/reweight the fields by before composing

    Returns:
       composed field
    """
    if isinstance(z_list, int):
        z_list = [z_list] * len(field_list)
    else:
        assert(len(z_list) == len(field_list))
    if isinstance(mip_list, int):
        mip_list = [mip_list] * len(field_list)
    else:
        assert(len(mip_list) == len(field_list))
    assert(min(mip_list) >= dst_mip)
    if factors is None:
        factors = [1.0] * len(field_list)
    else:
        assert(len(factors) == len(field_list))
    padded_bbox = deepcopy(bbox)
    print('Padding by {} at MIP{}'.format(pad, dst_mip))
    padded_bbox.uncrop(pad, mip=dst_mip)

    # load the first vector field
    f_cv, *field_list = field_list
    f_z, *z_list = z_list
    f_mip, *mip_list = mip_list
    f_factor, *factors = factors
    f = self.get_field(f_cv, f_z, padded_bbox, f_mip,
                       relative=False, to_tensor=True)
    f = f * f_factor

    # skip any empty / identity fields
    while is_identity(f):
        f_cv, *field_list = field_list
        f_z, *z_list = z_list
        f_mip, *mip_list = mip_list
        f_factor, *factors = factors
        f = self.get_field(f_cv, f_z, padded_bbox, f_mip,
                           relative=False, to_tensor=True)
        f = f * f_factor
        if len(field_list) == 0:
            return f

    if f_mip > dst_mip:
        f = upsample_field(f, f_mip, dst_mip)

    # compose with the remaining fields
    while len(field_list) > 0:
        g_cv, *field_list = field_list
        g_z, *z_list = z_list
        g_mip, *mip_list = mip_list
        g_factor, *factors = factors

        distance = self.profile_field(f)
        distance = (distance // (2 ** g_mip)) * 2 ** g_mip
        new_bbox = self.adjust_bbox(padded_bbox, distance.flip(0))

        f -= distance.to(device=self.device)
        f = self.abs_to_rel_residual(f, padded_bbox, dst_mip)
        f = f.to(device=self.device)

        g = self.get_field(g_cv, g_z, new_bbox, g_mip,
                           relative=False, to_tensor=True)
        g = g * g_factor
        if g_mip > dst_mip:
            g = upsample_field(g, g_mip, dst_mip)
        g = self.abs_to_rel_residual(g, padded_bbox, dst_mip)
        h = compose_fields(f, g)
        h = self.rel_to_abs_residual(h, dst_mip)
        h += distance.to(device=self.device)
        f = h

    f = f[:, pad:-pad, pad:-pad, :]
    return f

  def cloudsample_image_batch(self, z_range, image_cv, field_cv, 
                              bbox, image_mip, field_mip,
                              mask_cv=None, mask_mip=0, mask_val=0,
                              as_int16=True):
    """Warp a batch of sections using the cloudsampler 
   
    Args:
       z_range: list of ints for section indices to process
       image_cv: MiplessCloudVolume of source image
       field_cv: MiplesscloudVolume of vector field
       bbox: BoundingBox of output region
       image_mip: int for MIP of the source image
       field_mip: int for MIP of the vector field

    Returns:
       torch tensor of all images, concatenated along axis=0 
    """
    start = time()
    batch = []
    print("cloudsample_image_batch for z_range={0}".format(z_range))
    for z in z_range: 
      image = self.cloudsample_image(z, z, image_cv, field_cv, bbox, 
                                  image_mip, field_mip, 
                                  mask_cv=mask_cv, mask_mip=mask_mip, mask_val=mask_val, 
                                  as_int16=as_int16)
      batch.append(image)
    return torch.cat(batch, axis=0)

  def downsample(self, cv, z, bbox, mip):
    data = self.get_image(cv, z, bbox, mip, adjust_contrast=False, to_tensor=True)
    data = interpolate(data, scale_factor=0.5, mode='bilinear')
    return data.cpu().numpy()

  def cpc_chunk(self, src_cv, tgt_cv, src_z, tgt_z, bbox, src_mip, dst_mip, norm=True):
    """Calculate the chunked pearson r between two chunks

    Args:
       src_cv: MiplessCloudVolume of source image
       tgt_cv: MiplessCloudVolume of target image
       src_z: int z index of one section to compare
       tgt_z: int z index of other section to compare
       bbox: BoundingBox of region to process
       src_mip: int MIP level of input src & tgt images
       dst_mip: int MIP level of output image, will dictate the size of the chunks
        used for the pearson r

    Returns:
       img for bbox at dst_mip containing pearson r at each pixel for the chunks
       in src & tgt images at src_mip
    """
    print('Compute CPC for {4} at MIP{0} to MIP{1}, {2}<-({2},{3})'.format(src_mip, 
                                                                   dst_mip, src_z, 
                                                                   tgt_z, 
                                                                   bbox.__str__(mip=0)))
    scale_factor = 2**(dst_mip - src_mip)
    src = self.get_image(src_cv, src_z, bbox, src_mip, normalizer=None,
                         to_tensor=True)
    tgt = self.get_image(tgt_cv, tgt_z, bbox, src_mip, normalizer=None,
                         to_tensor=True)
    print('src.shape {}'.format(src.shape))
    print('tgt.shape {}'.format(tgt.shape))
    return cpc(src, tgt, scale_factor, norm=norm, device=self.device)

  ######################
  # Dataset operations #
  ######################
  def copy(self, cm, src_cv, dst_cv, src_z, dst_z, bbox, mip, is_field=False,
           mask_cv=None, mask_mip=0, mask_val=0, prefix='', return_iterator=False):
    """Copy one CloudVolume to another

    Args:
       cm: CloudManager that corresponds to the src_cv, tgt_cv, and field_cv
       model_path: str for relative path to ModelArchive
       src_z: int for section index of source image
       dst_z: int for section index of destination image
       src_cv: MiplessCloudVolume where source image is stored 
       dst_cv: MiplessCloudVolume where destination image will be stored 
       bbox: BoundingBox for region where source and target image will be loaded,
        and where the resulting vector field will be written
       mip: int for MIP level images will be loaded and field will be stored at
       field: bool indicating whether this is a field CloudVolume to copy
       mask_cv: MiplessCloudVolume where source mask is stored
       mask_mip: int for MIP level at which source mask is stored
       mask_val: int for pixel value in the mask that should be zero-filled
       prefix: str used to write "finished" files for each task 
        (only used for distributed)

    Returns:
       a list of CopyTasks
    """
    if prefix == '':
      prefix = '{}_{}'.format(mip, dst_z)

    class CopyTaskIterator():
        def __init__(self, cl, start, stop):
          self.chunklist = cl
          self.start = start
          self.stop = stop
        def __len__(self):
          return self.stop - self.start
        def __getitem__(self, slc):
          itr = deepcopy(self)
          itr.start = slc.start
          itr.stop = slc.stop
          return itr
        def __iter__(self):
          for i in range(self.start, self.stop):
            chunk = self.chunklist[i]
            yield tasks.CopyTask(src_cv, dst_cv, src_z, dst_z, chunk, mip,
                                 is_field, mask_cv, mask_mip, mask_val, prefix)

    chunks = self.break_into_chunks(bbox, cm.dst_chunk_sizes[mip],
                                    cm.dst_voxel_offsets[mip], mip=mip, 
                                    max_mip=cm.max_mip)
    if return_iterator:
        return CopyTaskIterator(chunks,0, len(chunks))
    #tq = GreenTaskQueue('deepalign_zhen')
    #tq.insert_all(ptasks, parallel=2)
    else:
        batch = []
        for chunk in chunks: 
          batch.append(tasks.CopyTask(src_cv, dst_cv, src_z, dst_z, chunk, mip, 
                                      is_field, mask_cv, mask_mip, mask_val, prefix))
        return batch
  
  def compute_field(self, cm, model_path, src_cv, tgt_cv, field_cv,
                    src_z, tgt_z, bbox, mip, pad=2048, src_mask_cv=None,
                    src_mask_mip=0, src_mask_val=0, tgt_mask_cv=None,
                    tgt_mask_mip=0, tgt_mask_val=0, prefix='',
                    return_iterator=False, prev_field_cv=None, prev_field_z=None,
                    prev_field_inverse=False):
    """Compute field to warp src section to tgt section 
  
    Args:
       cm: CloudManager that corresponds to the src_cv, tgt_cv, and field_cv
       model_path: str for relative path to ModelArchive
       src_cv: MiplessCloudVolume where source image to be loaded
       tgt_cv: MiplessCloudVolume where target image to be loaded
       field_cv: MiplessCloudVolume where output vector field will be written
       src_z: int for section index of source image
       tgt_z: int for section index of target image
       bbox: BoundingBox for region where source and target image will be loaded,
        and where the resulting vector field will be written
       mip: int for MIP level images will be loaded and field will be stored at
       pad: int for amount of padding to add to bbox before processing
       wait: bool indicating whether to wait for all tasks must finish before proceeding
       prefix: str used to write "finished" files for each task 
        (only used for distributed)
       prev_field_cv: MiplessCloudVolume where field prior is stored. Field will be used 
        to apply initial translation to target image. If None, will ignore.
       prev_field_z: int for section index of previous field
       prev_field_inverse: bool indicating whether the inverse of the previous field
        should be used.
       
    """
    start = time()
    chunks = self.break_into_chunks(bbox, cm.dst_chunk_sizes[mip],
                                    cm.dst_voxel_offsets[mip], mip=mip, 
                                    max_mip=cm.max_mip)
    if prefix == '':
      prefix = '{}_{}_{}'.format(mip, src_z, tgt_z)

    class ComputeFieldTaskIterator():
        def __init__(self, cl, start, stop):
          self.chunklist = cl
          self.start = start
          self.stop = stop
        def __len__(self):
          return self.stop - self.start
        def __getitem__(self, slc):
          itr = deepcopy(self)
          itr.start = slc.start
          itr.stop = slc.stop
          return itr
        def __iter__(self):
          for i in range(self.start, self.stop):
            chunk = self.chunklist[i]
            yield tasks.ComputeFieldTask(model_path, src_cv, tgt_cv, field_cv,
                                          src_z, tgt_z, chunk, mip, pad,
                                          src_mask_cv, src_mask_val, src_mask_mip, 
                                          tgt_mask_cv, tgt_mask_val, tgt_mask_mip, prefix,
                                          prev_field_cv, prev_field_z, prev_field_inverse)
    if return_iterator:
        return ComputeFieldTaskIterator(chunks,0, len(chunks))
    else:
        batch = []
        for chunk in chunks:
          batch.append(tasks.ComputeFieldTask(model_path, src_cv, tgt_cv, field_cv,
                                              src_z, tgt_z, chunk, mip, pad,
                                              src_mask_cv, src_mask_val, src_mask_mip, 
                                              tgt_mask_cv, tgt_mask_val, tgt_mask_mip, prefix,
                                              prev_field_cv, prev_field_z, prev_field_inverse))
        return batch
  
  def render(self, cm, src_cv, field_cv, dst_cv, src_z, field_z, dst_z, 
                   bbox, src_mip, field_mip, mask_cv=None, mask_mip=0, 
                   mask_val=0, affine=None, prefix='', use_cpu=False,
             return_iterator= False):
    """Warp image in src_cv by field in field_cv and save result to dst_cv

    Args:
       cm: CloudManager that corresponds to the src_cv, field_cv, & dst_cv
       src_cv: MiplessCloudVolume where source image is stored 
       field_cv: MiplessCloudVolume where vector field is stored 
       dst_cv: MiplessCloudVolume where destination image will be written 
       src_z: int for section index of source image
       field_z: int for section index of vector field 
       dst_z: int for section index of destination image
       bbox: BoundingBox for region where source and target image will be loaded,
        and where the resulting vector field will be written
       src_mip: int for MIP level of src images 
       field_mip: int for MIP level of vector field; field_mip >= src_mip
       mask_cv: MiplessCloudVolume where source mask is stored
       mask_mip: int for MIP level at which source mask is stored
       mask_val: int for pixel value in the mask that should be zero-filled
       wait: bool indicating whether to wait for all tasks must finish before proceeding
       affine: 2x3 ndarray for preconditioning affine to use (default: None means identity)
       prefix: str used to write "finished" files for each task 
        (only used for distributed)
    """
    start = time()
    chunks = self.break_into_chunks(bbox, cm.dst_chunk_sizes[src_mip],
                                    cm.dst_voxel_offsets[src_mip], mip=src_mip, 
                                    max_mip=cm.max_mip)
    if prefix == '':
      prefix = '{}_{}_{}'.format(src_mip, src_z, dst_z)
    class RenderTaskIterator():
        def __init__(self, cl, start, stop):
          self.chunklist = cl
          self.start = start
          self.stop = stop
        def __len__(self):
          return self.stop - self.start
        def __getitem__(self, slc):
          itr = deepcopy(self)
          itr.start = slc.start
          itr.stop = slc.stop
          return itr
        def __iter__(self):
          for i in range(self.start, self.stop):
            chunk = self.chunklist[i]
            yield tasks.RenderTask(src_cv, field_cv, dst_cv, src_z,
                       field_z, dst_z, chunk, src_mip, field_mip, mask_cv,
                       mask_mip, mask_val, affine, prefix, use_cpu)
    if return_iterator:
        return RenderTaskIterator(chunks,0, len(chunks))
    else:
        batch = []
        for chunk in chunks:
          batch.append(tasks.RenderTask(src_cv, field_cv, dst_cv, src_z,
                           field_z, dst_z, chunk, src_mip, field_mip, mask_cv,
                           mask_mip, mask_val, affine, prefix, use_cpu))
        return batch

  def vector_vote(self, cm, pairwise_cvs, vvote_cv, z, bbox, mip,
                  inverse=False, serial=True, prefix='', return_iterator=False,
                  softmin_temp=None, blur_sigma=None):
    """Compute consensus field from a set of vector fields

    Note: 
       tgt_z = src_z + z_offset

    Args:
       cm: CloudManager that corresponds to the src_cv, field_cv, & dst_cv
       pairwise_cvs: dict of MiplessCloudVolumes, indexed by their z_offset
       vvote_cv: MiplessCloudVolume where vector-voted field will be stored 
       z: int for section index to be vector voted 
       bbox: BoundingBox for region where all fields will be loaded/written
       mip: int for MIP level of fields
       inverse: bool indicating if pairwise fields are to be treated as inverse fields 
       serial: bool indicating to if a previously composed field is 
        not necessary
       wait: bool indicating whether to wait for all tasks must finish before proceeding
       prefix: str used to write "finished" files for each task 
        (only used for distributed)
    """
    start = time()
    chunks = self.break_into_chunks(bbox, cm.vec_chunk_sizes[mip],
                                    cm.vec_voxel_offsets[mip], mip=mip)
    if prefix == '':
      prefix = '{}_{}'.format(mip, z)
    class VvoteTaskIterator():
        def __init__(self, cl, start, stop):
          self.chunklist = cl
          self.start = start
          self.stop = stop
        def __len__(self):
          return self.stop - self.start
        def __getitem__(self, slc):
          itr = deepcopy(self)
          itr.start = slc.start
          itr.stop = slc.stop
          return itr
        def __iter__(self):
          for i in range(self.start, self.stop):
            chunk = self.chunklist[i]
            yield tasks.VectorVoteTask(deepcopy(pairwise_cvs), vvote_cv, z,
                                        chunk, mip, inverse, serial, prefix,
                                        softmin_temp=softmin_temp, blur_sigma=blur_sigma)
    if return_iterator:
        return VvoteTaskIterator(chunks,0, len(chunks))
    else:
        batch = []
        for chunk in chunks:
          batch.append(tasks.VectorVoteTask(deepcopy(pairwise_cvs), vvote_cv, z,
                                            chunk, mip, inverse, serial, prefix,
                                            softmin_temp=softmin_temp, 
                                            blur_sigma=blur_sigma))
        return batch

  def compose(self, cm, f_cv, g_cv, dst_cv, f_z, g_z, dst_z, bbox, 
                          f_mip, g_mip, dst_mip, factor, affine, pad, prefix='',
                          return_iterator=False):
    """Compose two vector field CloudVolumes

    For coarse + fine composition:
      f = fine 
      g = coarse 
    
    Args:
       cm: CloudManager that corresponds to the f_cv, g_cv, dst_cv
       f_cv: MiplessCloudVolume of vector field f
       g_cv: MiplessCloudVolume of vector field g
       dst_cv: MiplessCloudVolume of composed vector field
       f_z: int of section index to process
       g_z: int of section index to process
       dst_z: int of section index to process
       bbox: BoundingBox of region to process
       f_mip: MIP of vector field f
       g_mip: MIP of vector field g
       dst_mip: MIP of composed vector field
       affine: affine matrix
       pad: padding size
       prefix: str used to write "finished" files for each task 
        (only used for distributed)
    """
    start = time()
    chunks = self.break_into_chunks(bbox, cm.vec_chunk_sizes[dst_mip],
                                    cm.vec_voxel_offsets[dst_mip], 
                                    mip=dst_mip)
    if prefix == '':
      prefix = '{}_{}'.format(dst_mip, dst_z)

    class CloudComposeIterator():
        def __init__(self, cl, start, stop):
          self.chunklist = cl
          self.start = start
          self.stop = stop
        def __len__(self):
          return self.stop - self.start
        def __getitem__(self, slc):
          itr = deepcopy(self)
          itr.start = slc.start
          itr.stop = slc.stop
          return itr
        def __iter__(self):
          for i in range(self.start, self.stop):
            chunk = self.chunklist[i]
            yield tasks.CloudComposeTask(f_cv, g_cv, dst_cv, f_z, g_z, 
                                     dst_z, chunk, f_mip, g_mip, dst_mip,
                                     factor, affine, pad, prefix)
    if return_iterator:
        return CloudComposeIterator(chunks,0, len(chunks))
    else:
        batch = []
        for chunk in chunks:
          batch.append(tasks.CloudComposeTask(f_cv, g_cv, dst_cv, f_z, g_z, 
                                         dst_z, chunk, f_mip, g_mip, dst_mip,
                                         factor, affine, pad, prefix))
        return batch

  def multi_compose(self, cm, cv_list, dst_cv, z_list, dst_z, bbox, 
                                mip_list, dst_mip, factors, pad, prefix='',
                                return_iterator=False):
    """Compose a list of field CloudVolumes

    This takes a list of fields
    field_list = [f_0, f_1, ..., f_n]
    and composes them to get
    f_0 ⚬ f_1 ⚬ ... ⚬ f_n ~= f_0(f_1(...(f_n)))

    Args:
       cm: CloudManager that corresponds to the f_cv, g_cv, dst_cv
       cv_list: list of MiplessCloudVolume storing the vector fields
       dst_cv: MiplessCloudVolume of composed vector field
       z_list: int or list of ints for section indices to read fields
       dst_z: int of section index to process
       bbox: BoundingBox of region to process
       mip_list: int or list of ints for MIPs of the input fields
       dst_mip: MIP of composed vector field
       pad: padding size
       prefix: str used to write "finished" files for each task
        (only used for distributed)
    """
    chunks = self.break_into_chunks(bbox, cm.vec_chunk_sizes[dst_mip],
                                    cm.vec_voxel_offsets[dst_mip], 
                                    mip=dst_mip)
    if prefix == '':
        prefix = '{}_{}'.format(dst_mip, dst_z)

    if return_iterator:
        class CloudMultiComposeIterator():
            def __init__(self, cl, start, stop):
                self.chunklist = cl
                self.start = start
                self.stop = stop
            def __len__(self):
                return self.stop - self.start
            def __getitem__(self, slc):
                itr = deepcopy(self)
                itr.start = slc.start
                itr.stop = slc.stop
                return itr
            def __iter__(self):
                for i in range(self.start, self.stop):
                    chunk = self.chunklist[i]
                    yield tasks.CloudMultiComposeTask(cv_list, dst_cv, z_list,
                                                      dst_z, chunk, mip_list,
                                                      dst_mip, factors, pad,
                                                      prefix)
        return CloudMultiComposeIterator(chunks, 0, len(chunks))
    else:
        batch = []
        for chunk in chunks:
            batch.append(tasks.CloudMultiComposeTask(cv_list, dst_cv, z_list,
                                                dst_z, chunk, mip_list,
                                                dst_mip, factors, pad, prefix))
        return batch

  def cpc(self, cm, src_cv, tgt_cv, dst_cv, src_z, tgt_z, bbox, src_mip, dst_mip, 
                norm=True, prefix='', return_iterator=False):
    """Chunked Pearson Correlation between two CloudVolume images

    Args:
       cm: CloudManager that corresponds to the src_cv, tgt_cv, dst_cv
       src_cv: MiplessCloudVolume of source image
       tgt_cv: MiplessCloudVolume of target image
       dst_cv: MiplessCloudVolume of destination image
       src_z: int z index of one section to compare
       tgt_z: int z index of other section to compare
       bbox: BoundingBox of region to process
       src_mip: int MIP level of input src & tgt images
       dst_mip: int MIP level of output image, will dictate the size of the chunks
        used for the pearson r
       norm: bool for whether to normalize or not
       prefix: str used to write "finished" files for each task 
        (only used for distributed)
    """
    start = time()
    chunks = self.break_into_chunks(bbox, cm.vec_chunk_sizes[dst_mip],
                                    cm.vec_voxel_offsets[dst_mip], 
                                    mip=dst_mip)
    if prefix == '':
      prefix = '{}_{}'.format(dst_mip, src_z)

    class CpcIterator():
        def __init__(self, cl, start, stop):
          self.chunklist = cl
          self.start = start
          self.stop = stop
        def __len__(self):
          return self.stop - self.start
        def __getitem__(self, slc):
          itr = deepcopy(self)
          itr.start = slc.start
          itr.stop = slc.stop
          return itr
        def __iter__(self):
          for i in range(self.start, self.stop):
            chunk = self.chunklist[i]
            yield tasks.CPCTask(src_cv, tgt_cv, dst_cv, src_z, tgt_z, 
                                 chunk, src_mip, dst_mip, norm, prefix)
    if return_iterator:
        return CpcIterator(chunks,0, len(chunks))
    else:
        batch = []
        for chunk in chunks:
          batch.append(tasks.CPCTask(src_cv, tgt_cv, dst_cv, src_z, tgt_z, 
                                     chunk, src_mip, dst_mip, norm, prefix))
        return batch

  def render_batch_chunkwise(self, src_z, field_cv, field_z, dst_cv, dst_z, bbox, mip,
                   batch):
    """Chunkwise render

    Warp the image in BBOX at MIP and SRC_Z in CloudVolume dir at SRC_Z_OFFSET, 
    using the field at FIELD_Z in CloudVolume dir at FIELD_Z_OFFSET, and write 
    the result to DST_Z in CloudVolume dir at DST_Z_OFFSET. Chunk BBOX 
    appropriately.
    """
    
    print('Rendering src_z={0} @ MIP{1} to dst_z={2}'.format(src_z, mip, dst_z), flush=True)
    start = time()
    print("chunk_size: ", cm.dst_chunk_sizes[mip], cm.dst_voxel_offsets[mip])
    chunks = self.break_into_chunks_v2(bbox, cm.dst_chunk_sizes[mip],
                                    cm.dst_voxel_offsets[mip], mip=mip, render=True)
    if self.distributed:
        batch = []
        for i in range(0, len(chunks), self.task_batch_size):
            task_patches = []
            for j in range(i, min(len(chunks), i + self.task_batch_size)):
                task_patches.append(chunks[j].serialize())
            batch.append(tasks.RenderLowMipTask(src_z, field_cv, field_z, 
                                                           task_patches, image_mip, 
                                                           vector_mip, dst_cv, dst_z))
            self.upload_tasks(batch)
        self.wait_for_queue_empty(dst_cv.path, 'render_done/'+str(mip)+'_'+str(dst_z)+'/', len(chunks))
    else:
        def chunkwise(patch_bbox):
          warped_patch = self.cloudsample_image_batch(src_z, field_cv, field_z,
                                                      patch_bbox, mip, batch)
          self.save_image_batch(dst_cv, (dst_z, dst_z + batch), warped_patch, patch_bbox, mip)
        self.pool.map(chunkwise, chunks)
    end = time()
    print (": {} sec".format(end - start))

  def downsample_chunkwise(self, cv, z, bbox, source_mip, target_mip, wait=True):
    """Chunkwise downsample

    For the CloudVolume dirs at Z_OFFSET, warp the SRC_IMG using the FIELD for
    section Z in region BBOX at MIP. Chunk BBOX appropriately and save the result
    to DST_IMG.
    """
    print("Downsampling {} from mip {} to mip {}".format(bbox.__str__(mip=0), source_mip, target_mip))
    for m in range(source_mip+1, target_mip+1):
      chunks = self.break_into_chunks(bbox, cm.dst_chunk_sizes[m],
                                      cm.dst_voxel_offsets[m], mip=m, render=True)
      if self.distributed and len(chunks) > self.task_batch_size * 4:
          batch = []
          print("Distributed downsampling to mip", m, len(chunks)," chunks")
          for i in range(0, len(chunks), self.task_batch_size * 4):
              task_patches = []
              for j in range(i, min(len(chunks), i + self.task_batch_size * 4)):
                  task_patches.append(chunks[j].serialize())
              batch.append(tasks.DownsampleTask(cv, z, task_patches, mip=m))
          self.upload_tasks(batch)
          if wait:
            self.task_queue.block_until_empty()
      else:
          def chunkwise(patch_bbox):
            print ("Local downsampling {} to mip {}".format(patch_bbox.__str__(mip=0), m))
            downsampled_patch = self.downsample_patch(cv, z, patch_bbox, m-1)
            self.save_image_patch(cv, z, downsampled_patch, patch_bbox, m)
          self.pool.map(chunkwise, chunks)

  def render_section_all_mips(self, src_z, field_cv, field_z, dst_cv, dst_z, bbox, mip, wait=True):
    self.render(src_z, field_cv, field_z, dst_cv, dst_z, bbox, self.render_low_mip, wait=wait)
    # self.render_grid_cv(src_z, field_cv, field_z, dst_cv, dst_z, bbox, self.render_low_mip)
    self.downsample(dst_cv, dst_z, bbox, self.render_low_mip, self.render_high_mip, wait=wait)
  
  def render_to_low_mip(self, src_z, field_cv, field_z, dst_cv, dst_z, bbox, image_mip, vector_mip):
      self.low_mip_render(src_z, field_cv, field_z, dst_cv, dst_z, bbox, image_mip, vector_mip)
      self.downsample(dst_cv, dst_z, bbox, image_mip, self.render_high_mip)

  def compute_section_pair_residuals(self, src_z, src_cv, tgt_z, tgt_cv, field_cv,
                                     bbox, mip):
    """Chunkwise vector field inference for section pair

    Args:
       src_z: int for section index of source image
       src_cv: MiplessCloudVolume where source image to be loaded
       tgt_z: int for section index of target image
       tgt_cv: MiplessCloudVolume where target image to be loaded
       field_cv: MiplessCloudVolume where output vector field will be written
       bbox: BoundingBox for region where source and target image will be loaded,
        and where the resulting vector field will be written
       mip: int for MIP level images will be loaded and field will be stored at
    """
    start = time()
    chunks = self.break_into_chunks(bbox, self.dst[0].vec_chunk_sizes[mip],
                                    self.dst[0].vec_voxel_offsets[mip], mip=mip)
    print ("compute residuals between {} to slice {} at mip {} ({} chunks)".
           format(src_z, tgt_z, mip, len(chunks)), flush=True)
    if self.distributed:
      cv_path = self.dst[0].root
      batch = []
      for patch_bbox in chunks:
        batch.append(tasks.ResidualTask(src_z, src_cv, tgt_z, tgt_cv,
                                                  field_cv, patch_bbox, mip,
                                                  cv_path))
      self.upload_tasks(batch)
    else:
      def chunkwise(patch_bbox):
      #FIXME Torch runs out of memory
      #FIXME batchify download and upload
        self.compute_residual_patch(src_z, src_cv, tgt_z, tgt_cv,
                                    field_cv, patch_bbox, mip)
      self.pool.map(chunkwise, chunks)
    end = time()
    print (": {} sec".format(end - start))
    
  def count_box(self, bbox, mip):    
    chunks = self.break_into_chunks(bbox, self.dst[0].dst_chunk_sizes[mip],
                                      self.dst[0].dst_voxel_offsets[mip], mip=mip, render=True)
    total_chunks = len(chunks)
    self.image_pixels_sum =np.zeros(total_chunks)
    self.field_sf_sum =np.zeros((total_chunks, 2), dtype=np.float32)

  def invert_field_chunkwise(self, z, src_cv, dst_cv, bbox, mip, optimizer=False):
    """Chunked-processing of vector field inversion 
    
    Args:
       z: section of fields to weight
       src_cv: CloudVolume for forward field
       dst_cv: CloudVolume for inverted field
       bbox: boundingbox of region to process
       mip: field MIP level
       optimizer: bool to use the Optimizer instead of the net
    """
    start = time()
    chunks = self.break_into_chunks(bbox, cm.vec_chunk_sizes[mip],
                                    cm.vec_voxel_offsets[mip], mip=mip)
    print("Vector field inversion for slice {0} @ MIP{1} ({2} chunks)".
           format(z, mip, len(chunks)), flush=True)
    if self.distributed:
        batch = []
        for patch_bbox in chunks:
          batch.append(tasks.InvertFieldTask(z, src_cv, dst_cv, patch_bbox, 
                                                      mip, optimizer))
        self.upload_tasks(batch)
    else: 
    #for patch_bbox in chunks:
        def chunkwise(patch_bbox):
          self.invert_field(z, src_cv, dst_cv, patch_bbox, mip)
        self.pool.map(chunkwise, chunks)
    end = time()
    print (": {} sec".format(end - start))

  def res_and_compose(self, model_path, src_cv, tgt_cv, z, tgt_range, bbox,
                      mip, write_F_cv, pad, softmin_temp, prefix=""):
      T = 2**mip
      fields = []
      for z_offset in tgt_range:
          src_z = z
          tgt_z = src_z - z_offset
          print("calc res for src {} and tgt {}".format(src_z, tgt_z))
          f = self.compute_field_chunk(model_path, src_cv, tgt_cv, src_z,
                                       tgt_z, bbox, mip, pad)
          #print("--------f shape is ---", f.shape)
          fields.append(f)
          #fields.append(f)
      fields = [torch.from_numpy(i).to(device=self.device) for i in fields]
      #print("device is ", fields[0].device)
      field = vector_vote(fields, softmin_temp=softmin_temp)
      field = field.data.cpu().numpy()
      self.save_field(field, write_F_cv, z, bbox, mip, relative=False)

  def downsample_range(self, cv, z_range, bbox, source_mip, target_mip):
    """Downsample a range of sections, downsampling a given MIP across all sections
       before proceeding to the next higher MIP level.
    
    Args:
       cv: MiplessCloudVolume where images will be loaded and written
       z_range: list of ints for section indices that will be downsampled
       bbox: BoundingBox for region to be downsampled in each section
       source_mip: int for MIP level of the data to be initially loaded
       target_mip: int for MIP level after which downsampling will stop
    """
    for mip in range(source_mip, target_mip):
      print('downsample_range from {src} to {tgt}'.format(src=source_mip, tgt=target_mip))
      for z in z_range:
        self.downsample(cv, z, bbox, mip, mip+1, wait=False)
      if self.distributed:
        self.task_handler.wait_until_ready()
    

  def generate_pairwise_and_compose(self, z_range, compose_start, bbox, mip, forward_match,
                                    reverse_match, batch_size=1):
    """Create all pairwise matches for each SRC_Z in Z_RANGE to each TGT_Z in TGT_RADIUS
  
    Args:
        z_range: list of z indices to be matches 
        bbox: BoundingBox object for bounds of 2D region
        forward_match: bool indicating whether to match from z to z-i
          for i in range(tgt_radius)
        reverse_match: bool indicating whether to match from z to z+i
          for i in range(tgt_radius)
        batch_size: (for distributed only) int describing how many sections to issue 
          multi-match tasks for, before waiting for all tasks to complete
    """
    
    m = mip
    batch_count = 0
    start = 0
    chunks = self.break_into_chunks(bbox, cm.vec_chunk_sizes[m],
                                    cm.vec_voxel_offsets[m], mip=m)
    if forward_match:
      cm.add_composed_cv(compose_start, inverse=False,
                                  as_int16=as_int16)
      write_F_k = cm.get_composed_key(compose_start, inverse=False)
      write_F_cv = cm.for_write(write_F_k)
    if reverse_match:
      cm.add_composed_cv(compose_start, inverse=True,
                                  as_int16=as_int16)
      write_invF_k = cm.get_composed_key(compose_start, inverse=True)
      write_F_cv = cm.for_write(write_invF_k)

    for z in z_range:
      start = time()
      batch_count += 1
      i = 0
      if self.distributed:
        print("chunks size is", len(chunks))
        batch = []
        for patch_bbox in chunks:
            batch.append(tasks.ResAndComposeTask(z, forward_match,
                                                        reverse_match,
                                                        patch_bbox, mip,
                                                        write_F_cv))
        self.upload_tasks(batch)
      else:
        def chunkwise(patch_bbox):
            self.res_and_compose(z, forward_match, reverse_match, patch_bbox,
                                mip, write_F_cv)
        self.pool.map(chunkwise, chunks)
      if batch_count == batch_size and self.distributed:
        print('generate_pairwise waiting for {batch} sections'.format(batch=batch_size))
        print('batch_count is {}'.format(batch_count), flush = True)
        self.task_queue.block_until_empty()
        end = time()
        print (": {} sec".format(end - start))
        batch_count = 0
    # report on remaining sections after batch 
    if batch_count > 0 and self.distributed:
      print('generate_pairwise waiting for {batch} sections'.format(batch=batch_size))
      self.task_queue.block_until_empty()
      end = time()
      print (": {} sec".format(end - start))

  def compute_field_and_vector_vote(self, cm, model_path, src_cv, tgt_cv, vvote_field,
                          tgt_range, z, bbox, mip, pad, softmin_temp, prefix):
    """Create all pairwise matches for each SRC_Z in Z_RANGE to each TGT_Z in
    TGT_RADIUS and perform vetor voting
    """

    m = mip
    chunks = self.break_into_chunks(bbox, cm.vec_chunk_sizes[m],
                                    cm.vec_voxel_offsets[m], mip=m,
                                    max_mip=cm.num_scales)
    batch = []
    for patch_bbox in chunks:
        batch.append(tasks.ResAndComposeTask(model_path, src_cv, tgt_cv, z,
                                            tgt_range, patch_bbox, mip,
                                            vvote_field, pad, softmin_temp,
                                            prefix))
    return batch

  def generate_pairwise(self, z_range, bbox, forward_match, reverse_match, 
                              render_match=False, batch_size=1, wait=True):
    """Create all pairwise matches for each SRC_Z in Z_RANGE to each TGT_Z in TGT_RADIUS
  
    Args:
        z_range: list of z indices to be matches 
        bbox: BoundingBox object for bounds of 2D region
        forward_match: bool indicating whether to match from z to z-i
          for i in range(tgt_radius)
        reverse_match: bool indicating whether to match from z to z+i
          for i in range(tgt_radius)
        render_match: bool indicating whether to separately render out
          each aligned section before compiling vector fields with voting
          (useful for debugging)
        batch_size: (for distributed only) int describing how many sections to issue 
          multi-match tasks for, before waiting for all tasks to complete
        wait: (for distributed only) bool to wait after batch_size for all tasks
          to finish
    """
    
    mip = self.process_low_mip
    batch_count = 0
    start = 0
    for z in z_range:
      start = time()
      batch_count += 1 
      self.multi_match(z, forward_match=forward_match, reverse_match=reverse_match, 
                       render=render_match)
      if batch_count == batch_size and self.distributed and wait:
        print('generate_pairwise waiting for {batch} section(s)'.format(batch=batch_size))
        self.task_queue.block_until_empty()
        end = time()
        print (": {} sec".format(end - start))
        batch_count = 0
    # report on remaining sections after batch 
    if batch_count > 0 and self.distributed and wait:
      print('generate_pairwise waiting for {batch} section(s)'.format(batch=batch_size))
      self.task_queue.block_until_empty()
    end = time()
    print (": {} sec".format(end - start))
    #if self.p_render:
    #    self.task_queue.block_until_empty()
 
  def compose_pairwise(self, z_range, compose_start, bbox, mip,
                             forward_compose=True, inverse_compose=True, 
                             negative_offsets=False, serial_operation=False):
    """Combine pairwise vector fields in TGT_RADIUS using vector voting, while composing
    with earliest section at COMPOSE_START.

    Args
       z_range: list of ints (assumed to be monotonic & sequential)
       compose_start: int of earliest section used in composition
       bbox: BoundingBox defining chunk region
       mip: int for MIP level of data
       forward_compose: bool, indicating whether to compose with forward transforms
       inverse_compose: bool, indicating whether to compose with inverse transforms
       negative_offsets: bool indicating whether to use offsets less than 0 (z-i <-- z)
       serial_operation: bool indicating to if a previously composed field is 
        not necessary
    """
    
    T = 2**mip
    print('softmin temp: {0}'.format(T))
    if forward_compose:
      cm.add_composed_cv(compose_start, inverse=False,
                                  as_int16=as_int16)
    if inverse_compose:
      cm.add_composed_cv(compose_start, inverse=True,
                                  as_int16=as_int16)
    write_F_k = cm.get_composed_key(compose_start, inverse=False)
    write_invF_k = cm.get_composed_key(compose_start, inverse=True)
    read_F_k = write_F_k
    read_invF_k = write_invF_k
     
    if forward_compose:
      read_F_cv = cm.for_read(read_F_k)
      write_F_cv = cm.for_write(write_F_k)
      self.vector_vote_chunkwise(z_range, read_F_cv, write_F_cv, bbox, mip, 
                                 inverse=False, T=T, negative_offsets=negative_offsets,
                                 serial_operation=serial_operation)
    if inverse_compose:
      read_F_cv = cm.for_read(read_invF_k)
      write_F_cv = cm.for_write(write_invF_k)
      self.vector_vote_chunkwise(z_range, read_F_cv, write_F_cv, bbox, mip, 
                                 inverse=False, T=T, negative_offsets=negative_offsets,
                                 serial_operation=serial_operation)

  def get_neighborhood(self, z, F_cv, bbox, mip):
    """Compile all vector fields that warp neighborhood in TGT_RANGE to Z

    Args
       z: int for index of SRC section
       F_cv: CloudVolume with fields 
       bbox: BoundingBox defining chunk region
       mip: int for MIP level of data
    """
    fields = []
    z_range = [z+z_offset for z_offset in range(self.tgt_radius + 1)]
    for k, tgt_z in enumerate(z_range):
      F = self.get_field(F_cv, tgt_z, bbox, mip, relative=True, to_tensor=True,
                        as_int16=as_int16)
      fields.append(F)
    return torch.cat(fields, 0)
 
  def shift_neighborhood(self, Fs, z, F_cv, bbox, mip, keep_first=False): 
    """Shift field neighborhood by dropping earliest z & appending next z
  
    Args
       invFs: 4D torch tensor of inverse composed vector vote fields
       z: int representing the z of the input invFs. invFs will be shifted to z+1.
       F_cv: CloudVolume where next field will be loaded 
       bbox: BoundingBox representing xy extent of invFs
       mip: int for data resolution of the field
    """
    next_z = z + self.tgt_radius + 1
    next_F = self.get_field(F_cv, next_z, bbox, mip, relative=True,
                            to_tensor=True, as_int16=as_int16)
    if keep_first:
      return torch.cat((Fs, next_F), 0)
    else:
      return torch.cat((Fs[1:, ...], next_F), 0)

  def regularize_z(self, z_range, dir_z, bbox, mip, sigma=1.4):
    """For a given chunk, temporally regularize each Z in Z_RANGE
    
    Make Z_RANGE as large as possible to avoid IO: self.shift_field
    is called to add and remove the newest and oldest sections.

    Args
       z_range: list of ints (assumed to be a contiguous block)
       overlap: int for number of sections that overlap with a chunk
       bbox: BoundingBox defining chunk region
       mip: int for MIP level of data
       sigma: float standard deviation of the Gaussian kernel used for the
        weighted average inverse
    """
    block_size = len(z_range)
    overlap = self.tgt_radius
    curr_block = z_range[0]
    next_block = curr_block + block_size
    cm.add_composed_cv(curr_block, inverse=False,
                                as_int16=as_int16)
    cm.add_composed_cv(curr_block, inverse=True,
                                as_int16=as_int16)
    cm.add_composed_cv(next_block, inverse=False,
                                as_int16=as_int16)
    F_cv = cm.get_composed_cv(curr_block, inverse=False, for_read=True)
    invF_cv = cm.get_composed_cv(curr_block, inverse=True, for_read=True)
    next_cv = cm.get_composed_cv(next_block, inverse=False, for_read=False)
    z = z_range[0]
    invFs = self.get_neighborhood(z, invF_cv, bbox, mip)
    bump_dims = np.asarray(invFs.shape)
    bump_dims[0] = len(self.tgt_range)
    full_bump = create_field_bump(bump_dims, sigma)
    bump_z = 3 

    for z in z_range:
      composed = []
      bump = full_bump[bump_z:, ...]
      print(z)
      print(bump.shape)
      print(invFs.shape)
      F = self.get_field(F_cv, z, bbox, mip, relative=True, to_tensor=True,
                         as_int16=as_int16)
      avg_invF = torch.sum(torch.mul(bump, invFs), dim=0, keepdim=True)
      regF = compose_fields(avg_invF, F)
      regF = regF.data.cpu().numpy() 
      self.save_field(next_cv, z, regF, bbox, mip, relative=True, as_int16=as_int16)
      if z != z_range[-1]:
        invFs = self.shift_neighborhood(invFs, z, invF_cv, bbox, mip, 
                                        keep_first=bump_z > 0)
      bump_z = max(bump_z - 1, 0)

  def regularize_z_chunkwise(self, z_range, dir_z, bbox, mip, sigma=1.4):
    """Chunked-processing of temporal regularization 
    
    Args:
       z_range: int list, range of sections over which to regularize 
       dir_z: int indicating the z index of the CloudVolume dir
       bbox: BoundingBox of region to process
       mip: field MIP level
       sigma: float for std of the bump function 
    """
    start = time()
    # cm.add_composed_cv(compose_start, inverse=False)
    # cm.add_composed_cv(compose_start, inverse=True)
    chunks = self.break_into_chunks(bbox, cm.vec_chunk_sizes[mip],
                                    cm.vec_voxel_offsets[mip], mip=mip)
    print("Regularizing slice range {0} @ MIP{1} ({2} chunks)".
           format(z_range, mip, len(chunks)), flush=True)
    if self.distributed:
        batch = []
        for patch_bbox in chunks:
            batch.append(tasks.RegularizeTask(z_range[0], z_range[-1],
                                                      dir_z, patch_bbox,
                                                      mip, sigma))
        self.upload_tasks(batch)
        self.task_queue.block_until_empty()
    else:
        #for patch_bbox in chunks:
        def chunkwise(patch_bbox):
          self.regularize_z(z_range, dir_z, patch_bbox, mip, sigma=sigma)
        self.pool.map(chunkwise, chunks)
    end = time()
    print (": {} sec".format(end - start))

  def rechunck_image(self, chunk_size, image):
      I = image.split(chunk_size, dim=2)
      I = torch.cat(I, dim=0)
      I = I.split(chunk_size, dim=3)
      return torch.cat(I, dim=1)

  def calculate_fcorr(self, cm, bbox, mip, z1, z2, cv, dst_cv, dst_nopost, prefix=''):
      chunks = self.break_into_chunks(bbox, self.chunk_size,
                                      cm.dst_voxel_offsets[mip], mip=mip,
                                      max_mip=cm.max_mip)
      if prefix == '':
        prefix = '{}'.format(mip)
      batch = []
      for chunk in chunks:
        batch.append(tasks.ComputeFcorrTask(cv, dst_cv, dst_nopost, chunk, mip, z1, z2, prefix))
      return batch


  def get_fcorr(self, bbox, cv, mip, z1, z2):
      """ perform fcorr for two images
      """
      image1 = self.get_data(cv, z1, bbox, src_mip=mip, dst_mip=mip,
                             to_float=False, to_tensor=True).float()
      image2 = self.get_data(cv, z2, bbox, src_mip=mip, dst_mip=mip,
                             to_float=False, to_tensor=True).float()
      if(mip != 5):
        scale_factor = 2.**(mip - 5)
        image1 = interpolate(image1, scale_factor=scale_factor,
                             mode='bilinear')
        image2 = interpolate(image2, scale_factor=scale_factor,
                             mode='bilinear')
      std1 = image1[image1!=0].std()
      std2 = image2[image2!=0].std()
      scaling = 8 * pow(std1*std2, 1/2)
      fcorr_chunk_size = 8
      #print(image1)
      new_image1 = self.rechunck_image(fcorr_chunk_size, image1)
      new_image2 = self.rechunck_image(fcorr_chunk_size, image2)
      f1, p1 = get_fft_power2(new_image1)
      f2, p2 = get_fft_power2(new_image2)
      tmp_image = get_hp_fcorr(f1, p1, f2, p2, scaling=scaling)
      tmp_image = tmp_image.permute(2,3,0,1)
      tmp_image = tmp_image.cpu().numpy()
      tmp = deepcopy(tmp_image)
      tmp[tmp==2]=1
      blurred = scipy.ndimage.morphology.filters.gaussian_filter(tmp, sigma=(0, 0, 1, 1))
      s = scipy.ndimage.generate_binary_structure(2, 1)[None, None, :, :]
      closed = scipy.ndimage.morphology.grey_closing(blurred, footprint=s)
      closed = 2*closed
      closed[closed>1] = 1
      closed = 1-closed
      #print("++++closed shape",closed.shape)
      return closed, tmp_image

  def wait_for_queue_empty(self, path, prefix, chunks_len):
    if self.distributed:
      print("\nWait\n"
            "path {}\n"
            "prefix {}\n"
            "{} chunks\n".format(path, prefix, chunks_len), flush=True)
      empty = False
      n = 0
      while not empty:
        if n > 0:
          # sleep(1.75)
          sleep(5)
        with Storage(path) as stor:
            lst = stor.list_files(prefix=prefix)
        i = sum(1 for _ in lst)
        empty = (i == chunks_len)
        n += 1

  def wait_for_queue_empty_range(self, path, prefix, z_range, chunks_len):
      i = 0
      with Storage(path) as stor:
          for z in z_range:
              lst = stor.list_files(prefix=prefix+str(z))
              i += sum(1 for _ in lst)
      return i == chunks_len
  @retry
  def sqs_is_empty(self):
    # hashtag hackerlife
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
    self.sqs = boto3.client('sqs', region_name='us-east-1')
    self.queue_url  = self.sqs.get_queue_url(QueueName=self.queue_name)["QueueUrl"]
    print("\nSQS Wait")
    print("No. of messages / No. not visible")
    sleep(5)
    while not self.sqs_is_empty():
      sleep(1)
