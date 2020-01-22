import cv2
import argparse
import numpy as np
from cloudvolume import CloudVolume
from cloudvolume.lib import Bbox, Vec
import pandas as pd

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
    bslices = bbox.to_slices()
    src = cv[bslices[0], bslices[1], args.src_z][:,:,0,0].T
    tgt = cv[bslices[0], bslices[1], args.tgt_z][:,:,0,0].T

    print('Detecting SURF keypoints')
    detector = cv2.xfeatures2d_SURF.create(hessianThreshold=args.min_hessian)
    keypoints_src, descriptors_src = detector.detectAndCompute(src, None)
    keypoints_tgt, descriptors_tgt = detector.detectAndCompute(tgt, None)
    print('Found {} keypoints in src'.format(len(keypoints_src)))
    print('Found {} keypoints in tgt'.format(len(keypoints_tgt)))

    bf = cv2.BFMatcher()
    # matches = bf.match(descriptors_src, descriptors_tgt)
    # matches = bf.radiusMatch(descriptors_src, 
    #                               descriptors_tgt,
    #                               maxDistance=args.max_distance/2**args.mip)
    matches = bf.knnMatch(descriptors_src, descriptors_tgt, k=2)
    print('Found {} matches within radius {}'.format(len(matches), args.max_distance / 2**args.mip))
    points = []
    for m,n in matches:
        if (m.distance < 0.75*n.distance): 
            src_kp = keypoints_src[m.queryIdx]
            tgt_kp = keypoints_tgt[m.trainIdx]
            src_pt = np.array(src_kp.pt) 
            tgt_pt = np.array(tgt_kp.pt) 
            if np.linalg.norm(tgt_pt - src_pt) < args.max_distance / 2**args.mip:
                points.append([*src_kp.pt, *tgt_kp.pt])
    df = pd.DataFrame(data=np.array(points)*2**args.mip,
                      columns=['x0','y0','x1','y1'])
    df.loc[:,['x0','x1']] += bbox.minpt[0] * 2**args.mip
    df.loc[:,['y0','y1']] += bbox.minpt[1] * 2**args.mip
    df['z0'] = args.src_z
    df['z1'] = args.tgt_z
    df.to_csv(args.points_path, header=True, index=True)
