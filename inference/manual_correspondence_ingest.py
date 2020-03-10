import argparse
import os
import pandas as pd
from interpolate import sample_field 
from correspondences import load_points, save_points, determine_tgt
from features import feature_match
from getpass import getuser
from args import get_bbox
from aligner import Aligner
from render import render
from cloudmanager import CloudManager
from boundingbox import BoundingBox

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_points_path',
        type=str,
        help='path to manually subsampled points')
    parser.add_argument(
        '--output_points_path',
        type=str,
        default='',
        help='path to automatically subsampled points')
    parser.add_argument(
        '--src_path',
        type=str,
        help='path to CloudVolume with image')
    parser.add_argument(
        '--field_path',
        type=str,
        help='path to CloudVolume where to save interpolated grid') 
    parser.add_argument(
        '--dst_path',
        type=str,
        help='path to CloudVolume where to write render image')
    parser.add_argument(
        '--method',
        type=str,
        default='linear',
        help='interpolation option')
    parser.add_argument(
        '--src_mip',
        type=int,
        help='MIP level of image')
    parser.add_argument(
        '--field_mip',
        type=int,
        help='MIP level of interpolated grid')
    parser.add_argument(
        '--max_mip',
        type=int,
        default=13,
        help='MIP level of max desired grid')
    parser.add_argument(
        '--src_z',
        type=int,
        help='z index of source section')
    parser.add_argument(
        '--feature_match',
        action='store_true',
        help='compute SURF matches and combine with input matches')
    parser.add_argument(
        '--min_hessian',
        type=float,
        default=100,
        help='floor of Hessian used in SURF')
    parser.add_argument(
        '--max_distance',
        type=float,
        default=10000,
        help='maximum pixel distance of match allowed (at MIP0)')
    args = parser.parse_args()
    points_df = load_points(path=args.input_points_path, src_z=args.src_z)
    tgt_z = determine_tgt(points_df, args.src_z)
    provenance = {'description': 'Field created from manual correspondences & interpolation',
                  'sources': [args.src_path],
                  'owners': [getuser()],
                  'processing': [vars(args)]}
    cm = CloudManager(cv_path=args.src_path, 
                      max_mip=args.max_mip, 
                      max_displacement=0,
                      provenance=provenance, 
                      create_info=True)
    src = cm.create(path=args.src_path, 
                    data_type='uint8',
                    num_channels=1,
                    fill_missing=True, 
                    overwrite=False)
    field = cm.create(path=args.field_path, 
                    data_type='float32', 
                    num_channels=2,
                    fill_missing=True, 
                    overwrite=True)
    dst = cm.create(path=args.dst_path, 
                    data_type='uint8',
                    num_channels=1,
                    fill_missing=True, 
                    overwrite=True)
    bbox = BoundingBox.from_bbox(dst[args.field_mip].bounds, 
                                 mip=args.field_mip, 
                                 max_mip=args.field_mip)
    # match points
    if args.feature_match:
        print('Feature matching for src_z={} to tgt_z={}'.format(args.src_z, tgt_z))
        auto_df = feature_match(cv=src[args.src_mip], 
                            bbox=src[args.src_mip].bounds,
                            src_z=args.src_z, 
                            tgt_z=tgt_z, 
                            mip=args.src_mip, 
                            max_distance=args.max_distance, 
                            min_hessian=args.min_hessian)
        if len(auto_df) > 0:
            points_df = pd.concat((points_df, auto_df), axis=0, ignore_index=True)
        save_points(points_df, os.path.join(args.output_points_path,
                                            '{}-{}.csv'.format(args.src_z, tgt_z)))
    # interpolate points
    print('Interpolating for src_z={} to tgt_z={}'.format(args.src_z, tgt_z))
    displacements = sample_field(points_df=points_df,
                         bbox=bbox,
                         mip=args.field_mip,
                         method=args.method)
    a = Aligner()
    a.save_field(field=displacements,
                 cv=field,
                 z=args.src_z,
                 bbox=bbox,
                 mip=args.field_mip,
                 relative=False,
                 as_int16=False)
    # render image
    print('Rendering z={}'.format(args.src_z))
    render(a=a,
           cm=cm,
           src=src, 
           field=field, 
           dst=dst, 
           bbox=bbox,
           src_mip=args.src_mip,
           field_mip=args.field_mip,
           z_range=[args.src_z],
           queue_name=None)