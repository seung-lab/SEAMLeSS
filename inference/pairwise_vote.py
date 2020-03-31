import gevent.monkey
gevent.monkey.patch_all()

import argparse
from args import get_argparser, \
                 parse_args, \
                 get_aligner, \
                 get_bbox, \
                 get_provenance
from scheduler import Scheduler

from pairwisetensors import PairwiseTensors, PairwiseFields
from pairwisetensors import PairwiseVoteTask

if __name__ == '__main__':
    parser = get_argparser()
    parser.add_argument('--estimates_path',
        type=str,
        help='path to pairwise directory with estimated fields')
    parser.add_argument('--corrected_path',
        type=str,
        help='path to pairwise directory where corrected fields will be written')
    parser.add_argument('--weights_path',
        type=str,
        help='path to pairwise directory where field weights will be written')
    parser.add_argument('--offsets',
        type=int,
        nargs='+',
        help='list of pairwise offsets')
    parser.add_argument('--mip',
        type=int,
        help='int for MIP level of SRC_PATH')
    parser.add_argument('--max_mip',
        type=int,
        default=9,
        help='int for largest MIP level to have chunk alignment')
    parser.add_argument('--bbox_start',
        nargs=3,
        type=int,
        help='bbox origin, 3-element int list')
    parser.add_argument('--bbox_stop',
        nargs=3,
        type=int,
        help='bbox origin+shape, 3-element int list')
    parser.add_argument('--pad',
        default=2048,
        help='int for size of max displacement expected; should be 2^max_mip',
        type=int)
    parser.add_argument('--softmin_temp',
        default=1.,
        type=float,
        help='float for softmin temperature used in voting')
    parser.add_argument('--blur_sigma',
        default=1.,
        type=float,
        help='float for std of Gaussian used to blur fields ahead of voting')
    parser.add_argument('--cpu',
        action='store_true',
        help='set torch.device("cpu")')
    args = parse_args(parser)
    # a = get_aligner(args)
    print('Creating Scheduler')
    scheduler = Scheduler(queue_name=args.queue_name,
                            threads=args.threads)
    args.bbox_mip = 0
    bbox = get_bbox(args)
    mip = args.mip
    max_mip = args.max_mip
    provenance = get_provenance(args)
    pad = args.pad
    chunk_size = (1024, 1024)
    device = 'cpu' if args.cpu else 'cuda'
    offsets = args.offsets
    z_range = range(args.bbox_start[2], args.bbox_stop[2])
    softmin_temp = args.softmin_temp
    blur_sigma = args.blur_sigma

    print('Creating PairwiseFields & PairwiseTensor')
    # Test that estimates directory exists
    F = PairwiseFields(path=args.estimates_path,
                       offsets=offsets,
                       bbox=bbox,
                       mip=mip,
                       pad=pad,
                       device=device,
                       fill_missing=True)
    # F.exists() # NotImplemented
    field_info = F.info.copy()
    voxel_offset = field_info['scales'][mip]['voxel_offset']
    C = PairwiseFields(path=args.corrected_path,
                       offsets=offsets,
                       bbox=bbox,
                       mip=mip,
                       pad=pad,
                       device=device,
                       info=field_info,
                       fill_missing=True)
    C.mkdir()
    weight_info = F.info.copy()
    weight_info['num_channels'] = 1
    weight_info['data_type'] = 'float32'
    W = PairwiseTensors(path=args.weights_path,
                       offsets=offsets,
                       bbox=bbox,
                       mip=mip,
                       pad=pad,
                       device=device,
                       info=weight_info,
                       fill_missing=True)
    W.mkdir()

    class CorrectPairwiseFields(object):
        def __init__(self, z_range):
            self.z_range = z_range

        def __iter__(self):
            chunks = scheduler.get_chunks(bbox=bbox,
                                  chunk_size=chunk_size,
                                  voxel_offset=voxel_offset,
                                  mip=mip,
                                  max_mip=max_mip)
            for z in self.z_range:
                batch = []
                for chunk in chunks:
                    t = PairwiseVoteTask(estimates_path=args.estimates_path,
                                 corrected_path=args.corrected_path,
                                 weights_path=args.weights_path,
                                 offsets=offsets,
                                 src_z=z,
                                 tgt_offsets=offsets,
                                 bbox=chunk,
                                 mip=mip,
                                 pad=pad,
                                 device=device,
                                 softmin_temp=softmin_temp,
                                 blur_sigma=blur_sigma)
                    batch.append(t)
                yield from batch 

        def __repr__(self):
            return '{}({})'.format(self.__class__.__name__, self.z_range)

    print('CORRECT PAIRWISE FIELDS')
    scheduler.execute(CorrectPairwiseFields, z_range)
