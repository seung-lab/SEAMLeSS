import json
from pathlib import Path
from cloudfiles import CloudFiles
import argparse

def compile_levels(src_dir, z_start, z_stop, dst_path):
    """Compile levels counts from igneous LuminanceLevelsTask into single json

    Args:
        src_dir (str): CloudFiles path where luminancelevels output is stored per z
        z_start (int)
        z_stop (int)
        dst_path (str): local path where json will be written
    """
    cf = CloudFiles(src_dir, progress=False)
    levels = {}
    for z in range(z_start, z_stop):
        print(z, end='\r', flush=True)
        levels[z] = json.loads(cf.get(str(z)))['levels']
    with open(dst_path, 'w') as f:
        f.write(json.dumps(levels))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Compile LuminanceLevels into single json')
    parser.add_argument('--src_dir',
                        type=str,
                        help='Root dir of levels output')
    parser.add_argument('--dst_path',
                        type=str)
    parser.add_argument('--z_start',
                        type=int)
    parser.add_argument('--z_stop',
                        type=int)
    args = parser.parse_args()
    compile_levels(src_dir=args.src_dir,
                   z_start=args.z_start,
                   z_stop=args.z_stop,
                   dst_path=args.dst_path)