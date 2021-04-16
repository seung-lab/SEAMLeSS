import argparse
import copy
import numpy as np

def read_int_list_from_file(filename):
    int_list = []
    with open(filename, 'r') as f:
        line = f.readline()
        while line:
            int_list.append(int(line))
            line = f.readline()
    int_arr = np.array(int_list)
    return np.sort(int_arr)

def adjust_block_z(z, z_subs, minimum_block_size):
    any_subs = np.where(np.abs(z_subs - z) < minimum_block_size)[0]
    if len(any_subs) == 0:
        return z
    elif len(any_subs) == 1:
        return z_subs[any_subs[0]]
    else:
        raise ValueError(f'Ambiguous starter section subs {list(z_subs[any_subs])} for z={z}.')

def get_number_consecutive_missing(start_z, end_z, skip_arr):
    max_consecutive = 0
    first_missing_for_block_ind = np.searchsorted(skip_arr, start_z)
    last_missing_for_block_ind = np.searchsorted(skip_arr, end_z)
    if last_missing_for_block_ind > first_missing_for_block_ind:
        max_consecutive = 1
        cur_consecutive = 1
        cur_z = skip_arr[first_missing_for_block_ind]
        for i in range(first_missing_for_block_ind, last_missing_for_block_ind):
            if skip_arr[i] - 1 == cur_z:
                cur_consecutive = cur_consecutive + 1
                if cur_consecutive > max_consecutive:
                    max_consecutive = cur_consecutive
            else:
                cur_consecutive = 1
            cur_z = skip_arr[i]
    return max_consecutive

def calc_vv_num_for_block(start_z, end_z, skip_arr, begin_z, minimum_to_vector_vote, maximum_to_vector_vote):
    consecutive_missing = get_number_consecutive_missing(start_z, end_z, skip_arr)
    num_to_vector_vote = minimum_to_vector_vote + consecutive_missing
    if num_to_vector_vote % 2 == 0:
        num_to_vector_vote = num_to_vector_vote + 1
    while (start_z - num_to_vector_vote + 1) >= begin_z:
        consecutive_missing_start = get_number_consecutive_missing(start_z - num_to_vector_vote + 1, start_z, skip_arr)
        if consecutive_missing_start > (num_to_vector_vote - minimum_to_vector_vote):
            num_to_vector_vote = consecutive_missing_start + minimum_to_vector_vote
            if num_to_vector_vote % 2 == 0:
                num_to_vector_vote = num_to_vector_vote + 1
        else:
            break
    if num_to_vector_vote > maximum_to_vector_vote:
        return maximum_to_vector_vote
    return num_to_vector_vote

def generate_param_line(args, bbox_start, bbox_stop, vv_num):
    param_line_elements = [*bbox_start, *bbox_stop, 0, args.model_name, vv_num, 0, 'none']
    str_line_eles = [str(x) for x in param_line_elements]
    return ','.join(str_line_eles)

parser = argparse.ArgumentParser()
parser.add_argument("--dst_path", type=str)
parser.add_argument("--bbox_start", nargs=3, type=int)
parser.add_argument("--bbox_stop", nargs=3, type=int)
parser.add_argument("--alignment_z_start", type=int)
parser.add_argument("--alignment_block_size", type=int)
parser.add_argument("--minimum_to_vector_vote", type=int)
parser.add_argument("--maximum_to_vector_vote", type=int, default=11)
parser.add_argument("--model_name", type=str)
parser.add_argument("--starter_section_sub_path", type=str, default=None)
parser.add_argument("--skip_section_path", type=str, default=None)
args = parser.parse_args()
skip_z_arr = []
if args.skip_section_path is not None:
    skip_z_arr = read_int_list_from_file(args.skip_section_path)
starter_section_subs = []
if args.starter_section_sub_path is not None:
    starter_section_subs = read_int_list_from_file(args.starter_section_sub_path)
header_line = 'x_start,y_start,z_start,x_stop,y_stop,z_stop,mip,model_name,tgt_radius,skip,comment\n'
block_lines = [header_line]
minimum_block_size = 10
cur_bbox_start = args.bbox_start
cur_block_z = args.alignment_z_start
end_block_z_adjusted = None
was_last_block_adjusted = False
next_block_bbox_start = copy.deepcopy(args.bbox_start)
next_block_bbox_stop = copy.deepcopy(args.bbox_stop)
while cur_block_z < args.bbox_stop[2]:
    if end_block_z_adjusted is None:
        block_z_adjusted = adjust_block_z(cur_block_z, starter_section_subs, minimum_block_size)
    else:
        block_z_adjusted = end_block_z_adjusted
    block_end = min(cur_block_z + args.alignment_block_size, args.bbox_stop[2]+1)
    end_block_z_adjusted = adjust_block_z(block_end, starter_section_subs, minimum_block_size)
    vv_for_block = calc_vv_num_for_block(block_z_adjusted, end_block_z_adjusted, skip_z_arr, args.bbox_start[2], args.minimum_to_vector_vote, args.maximum_to_vector_vote)
    # print(f'vv = {vv_for_block}, bs = {block_z_adjusted}, be = {end_block_z_adjusted}')
    next_block_bbox_start[2] = block_z_adjusted
    next_block_bbox_stop[2] = end_block_z_adjusted
    block_lines.append(generate_param_line(args, next_block_bbox_start, next_block_bbox_stop, vv_for_block)+'\n')
    cur_block_z = block_end
with open(args.dst_path, 'w') as f:
    f.writelines(block_lines)
