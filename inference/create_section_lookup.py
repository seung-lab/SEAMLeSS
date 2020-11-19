import argparse
import json
import csv
from os.path import join
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--z_start', 
        type=int, 
        help='Start of range')
    parser.add_argument('--z_stop', 
        type=int, 
        help='End of range')
    parser.add_argument('--default_src_path', 
        type=str, 
        help='Default source path to assign unlisted sections')
    parser.add_argument('--src_paths', 
        nargs='+', 
        type=str, 
        help='Source paths to assign for listed sections')
    parser.add_argument('--z_list_paths', 
        nargs='+', 
        type=str, 
        help='Paths to files with list of sections corresponding to src_paths') 
    parser.add_argument('--dst_path', 
        type=str, 
        help='Path to section_lookup file')
    args = parser.parse_args()

    assert(len(args.z_list_paths) == len(args.src_paths))
    section_lookup = {}
    for src_path, z_list_path in zip(args.src_paths, args.z_list_paths):
        with open(z_list_path, 'r') as f:
            rows = csv.reader(f)
            for row in rows:
                z = int(row[0])
                if z not in section_lookup:
                    section = {}
                    section['z'] = z
                    section['src'] = src_path
                    section['transform'] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
                    section_lookup[z] = section

    for z in range(args.z_start, args.z_stop):
        if z not in section_lookup:
            section = {}
            section['z'] = z
            section['src'] = args.default_src_path
            section['transform'] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
            section_lookup[z] = section

    section_lookup = [v for v in section_lookup.values()]
    with open(args.dst_path, 'w') as f:
        json.dump(section_lookup, f, indent=4)
