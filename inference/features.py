import cv2
import argparse
import numpy as np
from cloudvolume import CloudVolume
from cloudvolume.lib import Bbox, Vec
import pandas as pd

def feature_match(cv, bbox, src_z, tgt_z, mip, max_distance, min_hessian):
    """Return matched SURF keypoints

    Args:
        cv: CloudVolume with images
        bbox: Bbox for xy extent
        src_z: int for section that will be warped
        tgt_z: int for section to compare against
        mip: int for MIP level of images
        max_distance: number of max distance between keypoints
        min_hessian: number for minimum of hessian determinant in SURF
    
    Returns
        pd.DataFrame with correpsondences as x,y,z pairs
    """
    bslices = bbox.to_slices()
    src = cv[bslices[0], bslices[1], src_z][:,:,0,0].T
    tgt = cv[bslices[0], bslices[1], tgt_z][:,:,0,0].T

    print('Detecting SURF keypoints')
    detector = cv2.xfeatures2d_SURF.create(hessianThreshold=min_hessian)
    keypoints_src, descriptors_src = detector.detectAndCompute(src, None)
    keypoints_tgt, descriptors_tgt = detector.detectAndCompute(tgt, None)
    print('Found {} keypoints in src'.format(len(keypoints_src)))
    print('Found {} keypoints in tgt'.format(len(keypoints_tgt)))

    bf = cv2.BFMatcher()
    # matches = bf.match(descriptors_src, descriptors_tgt)
    # matches = bf.radiusMatch(descriptors_src, 
    #                               descriptors_tgt,
    #                               maxDistance=max_distance/2**mip)
    matches = bf.knnMatch(descriptors_src, descriptors_tgt, k=2)
    print('Found {} matches within radius {}'.format(len(matches), max_distance / 2**mip))
    points = []
    for m,n in matches:
        if (m.distance < 0.75*n.distance): 
            src_kp = keypoints_src[m.queryIdx]
            tgt_kp = keypoints_tgt[m.trainIdx]
            src_pt = np.array(src_kp.pt) 
            tgt_pt = np.array(tgt_kp.pt) 
            if np.linalg.norm(tgt_pt - src_pt) < max_distance / 2**mip:
                points.append([*src_kp.pt, *tgt_kp.pt])
    cols = ['x0','y0','x1','y1']
    if len(points) > 0:
        df = pd.DataFrame(data=np.array(points)*2**mip,
                          columns=cols)
        df.loc[:,['x0','x1']] += bbox.minpt[0] * 2**mip
        df.loc[:,['y0','y1']] += bbox.minpt[1] * 2**mip
        df['z0'] = src_z
        df['z1'] = tgt_z
    else:
        df = pd.DataFrame(columns=cols)
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path',
        type=str,
        help='CloudVolume path')
    parser.add_argument(
        '--mip',
        type=int,
        help='MIP level of images')
    parser.add_argument(
        '--src_z',
        type=int,
        help='z index of source image')
    parser.add_argument(
        '--tgt_z',
        type=int,
        help='z index of target image')
    parser.add_argument(
        '--points_path',
        type=str,
        help='path for points csv to be written')
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
    parser.add_argument(
        '--bbox_start', 
        nargs=3, 
        type=int,
        help='bbox origin, 3-element int list')
    parser.add_argument(
        '--bbox_stop', 
        nargs=3, 
        type=int,
        help='bbox origin+shape, 3-element int list')
    args = parser.parse_args()
    cv = CloudVolume(args.path, mip=args.mip, fill_missing=True)
    bbox = cv.bounds
    if args.bbox_start is not None:
        bbox = Bbox(Vec(*args.bbox_start), Vec(*args.bbox_stop))
        bbox = cv.bbox_to_mip(bbox, mip=0, to_mip=args.mip)
    df = feature_match(cv=cv, 
                        bbox=bbox, 
                        src_z=args.src_z, 
                        tgt_z=args.tgt_z, 
                        mip=args.mip, 
                        max_distance=args.max_distance, 
                        min_hessian=args.min_hessian)
    df.to_csv(args.points_path, header=True, index=True)
