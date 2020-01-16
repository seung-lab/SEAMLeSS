import argparse
from scipy.interpolate import griddata
import numpy as np
import pandas as pd

class SubsampledField():

    def __init__(self, df):
        self.v = df 

    def compute_delta(self):
        for i in ['x','y']:
            self.v['d{}'.format(i)] = self.v['{}1'.format(i)] - self.v['{}0'.format(i)]

    def interpolate(self, grid_x, grid_y, mip=0, method='cubic'):
        scale = 2**mip
        df = self.v / scale
        x_interp = griddata(points=df[['x0','y0']].to_numpy(),
                            values=df['dx'].to_numpy(),
                            xi=(grid_x, grid_y),
                            method=method)
        y_interp = griddata(points=df[['x0','y0']].to_numpy(),
                            values=df['dy'].to_numpy(),
                            xi=(grid_x, grid_y),
                            method=method)
        if method != 'nearest':
            x_near = griddata(points=df[['x0','y0']].to_numpy(),
                                values=df['dx'].to_numpy(),
                                xi=(grid_x, grid_y),
                                method='nearest')
            y_near = griddata(points=df[['x0','y0']].to_numpy(),
                                values=df['dy'].to_numpy(),
                                xi=(grid_x, grid_y),
                                method='nearest')
            x_interp[np.isnan(x_interp)] = x_near[np.isnan(x_interp)]
            y_interp[np.isnan(y_interp)] = y_near[np.isnan(y_interp)]
        return np.stack((x_interp, y_interp), axis=-1)

def grid_to_points(grid_v, grid_x, grid_y):
    """Reduce field grid to list of points

    Args:
        grid_v: WxHx2 np.array with x,y components in last dim
        grid_x: WxH np.array with x values
        grid_y: WxH np.array with y values
    """
    data = [] 
    for i in range(grid_v.shape[0]): 
        for j in range(grid_v.shape[1]):
            x = grid_x[i,j]
            y = grid_y[i,j]
            v = grid_v[i,j,:]
            data.append([x,y,x+v[0],y+v[1]])
    return data 

def make_grid(x_start, x_stop, y_start, y_stop, mip=0):
    s = 2**mip
    x_start = int(np.floor(x_start/s))
    y_start = int(np.floor(y_start/s))
    x_stop = int(np.ceil(x_stop/s))
    y_stop = int(np.ceil(y_stop/s))
    return np.mgrid[x_start:x_stop, y_start:y_stop]
    
def add_z(points, z0, z1):
    return [[*p, z0, z1] for p in points] 

def test_subsample():
    df = pd.DataFrame(data=np.array([[0,0,2,0],
                                     [2,0,2,2],
                                     [2,2,0,2],
                                     [0,2,0,0]]),
                       columns=['x0','y0','x1','y1'])
    mip = 8
    s = 2**mip
    df = df * s
    f = SubsampledField(df)
    f.compute_delta()
    grid_x, grid_y = make_grid(-1*s,4*s,-1*s,4*s,mip=mip)
    grid_v = f.interpolate(grid_x, grid_y, mip=mip, method='linear')
    points = grid_to_points(grid_v.astype(int), grid_x, grid_y)
    idf = pd.DataFrame(data=points, 
                       columns=['x0','y0','x1','y1'])
    idf = idf * s 
    isf = SubsampledField(idf)

if __name__ == '__main__':
    # test_subsample()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--src_path',
        type=str,
        help='path to subsampled points')
    parser.add_argument(
        '--dst_path',
        type=str,
        help='path of where to save interpolated grid') 
    parser.add_argument(
        '--method',
        type=str,
        default='cubic',
        help='interpolation option')
    parser.add_argument(
        '--mip',
        type=int,
        help='MIP level of interpolated grid')
    parser.add_argument(
        '--bbox',
        nargs=6,
        type=int,
        help='interpolated grid bound (x_start, x_stop, y_start, y_stop, z_start, z_stop)')
    args = parser.parse_args()
    src_df = pd.read_csv(args.src_path, index_col=0)
    f = SubsampledField(src_df)
    f.compute_delta()
    grid_x, grid_y = make_grid(*args.bbox[:4], mip=args.mip)
    grid_v = f.interpolate(grid_x, grid_y, mip=args.mip, method=args.method)
    points = grid_to_points(grid_v, grid_x, grid_y)
    df = pd.DataFrame(data=np.array(points)*2**args.mip,
                       columns=['x0','y0','x1','y1'])
    df['z0'] = args.bbox[4]
    df['z1'] = args.bbox[5]
    df.to_csv(args.dst_path, header=True, index=True)
