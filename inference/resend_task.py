
from concurrent.futures import ProcessPoolExecutor
import taskqueue
from taskqueue import TaskQueue, GreenTaskQueue, LocalTaskQueue, MockTaskQueue

import sys
import torch
import json
import math
import csv
from copy import deepcopy
from time import time, sleep
from as_args import get_argparser, parse_args, get_aligner, get_bbox, get_provenance
from os.path import join
from cloudmanager import CloudManager
from itertools import compress
from tasks import run, remote_upload
from boundingbox import BoundingBox
from boundingbox import deserialize_bbox
import numpy as np
import os
from cloudvolume import Storage

tmp_dir= "/tmp/alignment/"

def print_run(diff, n_tasks):
  if n_tasks > 0:
    print (": {:.3f} s, {} tasks, {:.3f} s/tasks".format(diff, n_tasks, diff / n_tasks))

def calc_start_z(block_starts, block_stops, task_finish_dir, slice_finish_dir,
                skip_list):
    with Storage(task_finish_dir) as stor:
        lst = stor.list_files()
    finish_list = [int(i) for i in lst]
    bs_list = []
    be_list = []
    start_list =[]
    for i in range(len(block_starts)):
        bs = block_starts[i]
        if bs in finish_list:
            continue
        else:
            #print("i is", i)
            bs_list.append(bs)
            be_list.append(block_stops[i])
            with Storage(slice_finish_dir+str(bs)+'/') as stor:
                slice_list = stor.list_files()
            slice_list = [int(i) for i in slice_list]
            if len(slice_list) == 0:
                start_list.append(-1)
                continue
            number_list = list(map(int, slice_list))
            number_list.sort()
            start_z = -2
            for i, z in enumerate(number_list[:-1]):
                if (z+1 != number_list[i+1]) and (z+1 not in skip_list):
                    start_z = z+1
                    break
            if start_z == -2:
                start_z = number_list[-1]+1
            while start_z in skip_list:
                start_z += 1
            start_list.append(start_z)
    return bs_list, be_list, start_list

def remote_upload_it(tasks):
      with GreenTaskQueue(queue_name=args.queue_name) as tq:
          tq.insert_all(tasks)

def get_skip_list():
    file_name = tmp_dir+"skip"
    if os.path.exists(file_name):
        f = open(file_name,"r")
        skip_list = [int(i) for i in f.readlines()]
        return skip_list
    else:
        return -1

def find_non_consecutive(file_name, skip_list):
    number_list = os.listdir(file_name)
    if len(number_list) ==0:
       return -1
    number_list = list(map(int, number_list))
    number_list.sort()
    for i, z in enumerate(number_list[:-1]):
        if (z+1 != number_list[i+1]) and (z+1 not in skip_list):
            return z+1
    restart = number_list[-1]+1
    while restart in skip_list:
        restart += 1
    return restart

def get_task(a):
    skip_list = get_skip_list()
    if isinstance(skip_list, list):
        restart_z = find_non_consecutive(tmp_dir+"img", skip_list)
    else:
        restart_z = -1
    with open(tmp_dir+"name") as load_file:
        arg_dic = json.load(load_file)
       # file_reader = csv.reader(csv_file, delimiter=',')
       # for row in file_reader:
       #     arg_dic[row[0]] = row[1]
    #print("arg_dic is ", arg_dic)
    if arg_dic["class"] == "NewAlignTask":
        bs = int(arg_dic["block_start"])
        be = int(arg_dic["block_stop"])
        if restart_z ==-1:
            start_z = int(arg_dic["start_z"])
        else:
            start_z = int(restart_z)
        src = arg_dic["src"]
        dst = arg_dic["dst"]
        block_pair_field = arg_dic["s_field"]
        block_vvote_field = arg_dic["vvote_field"]
        chunk_grid =[deserialize_bbox(arg_dic["chunk_grid"][0])]
        #print(chunk_grid, type(chunk_grid))
        mip = int(arg_dic["mip"])
        pad = int(arg_dic["pad"])
        chunk_size = int(arg_dic["chunk_size"])
        param_lookup = arg_dic["model_lookup"]
        qu = arg_dic["qu"]
        mask_cv = arg_dic["mask_cv"]
        mask_mip = arg_dic["mask_mip"]
        mask_val = arg_dic["mask_val"]
        finish_dir = arg_dic["finish_dir"]
        t = a.new_align_task(bs, be, start_z, src, dst,
                             block_pair_field,
                             block_vvote_field,
                             chunk_grid, mip, pad,
                             chunk_size, param_lookup,
                             qu, finish_dir,
                             src_mask_cv=mask_cv,
                             src_mask_mip=mask_mip,
                             src_mask_val=mask_val)
    elif arg_dic["class"] == "StitchComposeRender":
        qu =arg_dic["qu"]
        if restart_z ==-1:
            start_z = arg_dic["z_start"]
        else:
            start_z = restart_z
        print("start_z is ", start_z)
        end_z = arg_dic["z_stop"]
        bbox = deserialize_bbox(arg_dic["bbox"])
        src = arg_dic["src"]
        dst = arg_dic["dst"]
        broadcasting_field = arg_dic["b_field"]
        block_field = arg_dic["vv_field_cv"]
        decay_dist = arg_dic["decay_dist"]
        influence_block = arg_dic["influence_blocks"]
        src_mip = arg_dic["src_mip"]
        dst_mip = arg_dic["dst_mip"]
        pad = arg_dic["pad"]
        extra_off = arg_dic["extra_off"]
        chunk_size = arg_dic["chunk_size"]
        upsample_mip = arg_dic["upsample_mip"]
        finish_dir = arg_dic["finish_dir"]
        influence_index = arg_dic["influence_index"]
        upsample_bbox = deserialize_bbox(arg_dic["upsample_bbox"])
        t = a.stitch_compose_render_task(qu, bbox, src, dst, influence_index,
                                         start_z, end_z, broadcasting_field,
                                         block_field, decay_dist,
                                         influence_block, finish_dir,
                                         src_mip, dst_mip, pad, pad,
                                         chunk_size,
                                         upsample_mip, upsample_bbox)
    elif arg_dic["class"] == "StitchGetField":
        qu = arg_dic["qu"]
        src_cv = arg_dic["src_cv"]
        tgt_cv = arg_dic["tgt_cv"]
        param_lookup = arg_dic["param_lookup"]
        block_vvote_field = arg_dic["prev_field_cv"]
        broadcasting_field = arg_dic["bfield_cv"]
        tmp_img_cv = arg_dic["tmp_img_cv"]
        tmp_vvote_field_cv = arg_dic["tmp_vvote_field_cv"]
        mip = int(arg_dic["mip"])
        bs = int(arg_dic["bs"])
        be = int(arg_dic["be"])
        finish_dir = arg_dic["finish_dir"]
        if restart_z ==-1:
            start_z = int(arg_dic["start_z"])
        else:
            start_z = int(restart_z)
        bbox = deserialize_bbox(arg_dic["bbox"])
        chunk_size = int(arg_dic["chunk_size"])
        pad = int(arg_dic["pad"])
        softmin_temp = arg_dic["softmin_temp"]
        blur_sigma = arg_dic["blur_sigma"]
        t = a.stitch_get_field_task_generator(qu, param_lookup,bs, be, src_cv, tgt_cv,
                                              block_vvote_field, broadcasting_field,
                                              tmp_img_cv,tmp_vvote_field_cv,
                                              mip, start_z, bbox, chunk_size,
                                              pad, finish_dir,
                                              softmin_temp, blur_sigma)
    return t


if __name__ == '__main__':
    parser = get_argparser()
    args = parse_args(parser)
    a = get_aligner(args)
    provenance = get_provenance(args)
    t = get_task(a)
    remote_upload(args.queue_name, t)
    a.wait_for_sqs_empty()

#  # Task scheduling functions
#  def remote_upload_it(tasks):
#      with GreenTaskQueue(queue_name=args.queue_name) as tq:
#          tq.insert_all(tasks)
#
#  qu = args.queue_name
#  start_z = -1
#
#  #print("z_range is ", z_range)
#  ptask = []
#  bs_list = make_range(block_starts, a.threads)
#  be_list = make_range(block_stops, a.threads)
#  for bs, be in zip(bs_list, be_list):
#      ptask.append(AlignT(bs, be))
#
#  with ProcessPoolExecutor(max_workers=a.threads) as executor:
#      executor.map(remote_upload_it, ptask)
#  start = time()
#  #print("start until now time", start - begin_time)
#  #a.wait_for_queue_empty(dst.path, 'load_image_done/{}'.format(mip), len(batch))
#  a.wait_for_sqs_empty()
#  end = time()
#  diff = end - start 
#  print("Executing Loading Tasks use time:", diff)
#
#
#  class StitchGetFieldT(object):
#      def __init__(self, bs_list, be_list):
#          self.bs_list = bs_list
#          self.be_list = be_list
#          print("*********self bs_list is ", self.bs_list)
#          print("*********be_list is  ", self.be_list)
#      def __iter__(self):
#          for bs, be in zip(self.bs_list, self.be_list):
#              even_odd = block_dst_lookup[bs]
#              src_cv = block_dsts[even_odd]
#              tgt_cv = block_dsts[(even_odd+1)%2]
#              t = a.stitch_get_field_task_generator(args.param_lookup,[bs],
#                                                    [be], src_cv, tgt_cv,
#                                                    block_vvote_field,
#                                                    broadcasting_field,
#                                                    src, mip,
#                                                    chunk_grid[0],
#                                                    chunk_size, pad, 2**mip, 1)
#              yield from t
#
#  #print("z_range is ", z_range)
#  ptask = []
#  bs_list = make_range(block_starts[1:], a.threads)
#  be_list = make_range(block_stops[1:], a.threads)
#  print("bs_list is ", bs_list)
#  print("be_list is ", be_list)
#  for bs, be in zip(bs_list, be_list):
#      ptask.append(StitchGetFieldT(bs,be))
#  #print("ptask len is ", len(ptask))
#  #ptask.append(batch)
#  #remote_upload_it(ptask)
#  with ProcessPoolExecutor(max_workers=a.threads) as executor:
#      executor.map(remote_upload_it, ptask)
#
#  start = time()
#  #print("start until now time", start - begin_time)
#  #a.wait_for_queue_empty(dst.path, 'load_image_done/{}'.format(mip), len(batch))
#  a.wait_for_sqs_empty()
#  end = time()
#  diff = end - start 
#  print("Executing Loading Tasks use time:", diff)
#
