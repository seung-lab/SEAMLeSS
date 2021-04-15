import argparse

def read_int_list_from_file(filename):
    int_list = []
    with open(filename, 'r') as f:
        line = f.readline()
        while line:
            int_list.append(int(line))
            line = f.readline()
    return int_list

parser = argparse.ArgumentParser()
parser.add_argument("--dst_path", type=str)
parser.add_argument("--bbox_start", nargs=3, type=int)
parser.add_argument("--bbox_stop", nargs=3, type=int)
parser.add_argument("--alignment_z_start", type=int)
parser.add_argument("--alignment_block_size", type=int)
parser.add_argument("--minimum_to_vector_vote", type=int)
parser.add_argument("--model_name", type=str)
parser.add_argument("--starter_section_sub_path", type=str, default=None)
args = parser.parse_args()
starter_section_subs = []
if args.starter_section_sub_path is not None:
    starter_section_subs = read_int_list_from_file(args.starter_section_sub_path)
minimum_block_size = 10
