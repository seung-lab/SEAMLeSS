import neuroglancer
import webbrowser
import argparse
import numpy as np
import pandas as pd
import sys

def parse_df(df):
    """Given CSV from neuroglancer export, extract and clean points
    """
    c1 = 'Coordinate 1'
    c2 = 'Coordinate 2'
    coord_cols = ['x0','y0','z0','x1','y1','z1']
    exported = True
    exported = c1 in df.columns
    if c2 not in df.columns:
        c2 = 'Coordinate 2 (if applicable)'
    exported = exported and c2 in df.columns
    if exported:
        df[['x0','y0','z0']] = df[c1].str.replace('(','') \
                                     .str.replace(')','') \
                                     .str.split(', ', expand=True) \
                                     .astype(int) 
        df[['x1','y1','z1']] = df[c2].str.replace('(','') \
                                     .str.replace(')','') \
                                     .str.split(', ', expand=True) \
                                     .astype(int) 
    for coord_col in coord_cols:
        assert(coord_col in df.columns)
    return df[coord_cols]

def filter_df(df, src_z, tgt_z):
    return df[df['z0'].isin([src_z, tgt_z]) & df['z1'].isin([src_z, tgt_z])]

def determine_tgt(df, src_z):
    """Given df with x,y,z pairs, determine most frequent tgt_z
    """
    values, counts = np.unique(df[['z0','z1']], return_counts=True)
    tgt_values, tgt_counts = values[values != src_z], counts[values != src_z]
    return tgt_values[np.argmax(tgt_counts)]

def sort_df(df, src_z):
    """Given df with x,y,z pairs, sort columns by z
    """
    swap = df['z0'] != src_z
    df.loc[swap, ['x0','y0','z0',
                  'x1','y1','z1']] = df.loc[swap, ['x1','y1','z1',
                                                   'x0','y0','z0']].values

def load_points(path, src_z):
    """Given filepath to points CSV, return pd.DataFrame with clean x,y,z pairs
    """
    df = pd.read_csv(path)
    df = parse_df(df)
    tgt_z = determine_tgt(df, src_z)
    df = filter_df(df, src_z, tgt_z)
    sort_df(df, src_z)
    return df

def save_points(df, path):
    columns=['Coordinate 1',
             'Coordinate 2',
             'Ellipsoid Dimensions',
             'Tags',
             'Description',
             'Segment IDs',
             'Parent ID',
             'Type',
             'ID']
    ngdf = pd.DataFrame(columns=columns)
    coord_map = lambda x: '({:.0f}, {:.0f}, {:.0f})'.format(*x)
    ngdf['Coordinate 1'] = df[['x0','y0','z0']].apply(coord_map, axis=1) 
    ngdf['Coordinate 2'] = df[['x1','y1','z1']].apply(coord_map, axis=1) 
    ngdf['Type'] = 'Line'
    ngdf.to_csv(path, header=True, index=False)


class CorrespondencesController:
    """Simple class to set & get point-pairs from neuroglancer
    """

    def __init__(self, source, bind_address=None, static_content_url=None, coords=None, **kwargs):
        if bind_address:
            neuroglancer.set_server_bind_address(bind_address)
        if static_content_url:
            neuroglancer.set_static_content_source(url=static_content_url)
        self.viewer = neuroglancer.Viewer()
        with self.viewer.txn() as state:
            state.layers['image'] = neuroglancer.ImageLayer(source=source)
        with self.viewer.txn() as state:
            state.layers["correspondences"] = neuroglancer.AnnotationLayer()
        with self.viewer.txn() as state:
            state.layout = neuroglancer.DataPanelLayout('xy')
        if coords:
            with self.viewer.txn() as state:
                state.voxel_coordinates = coords
                state.navigation.zoom_factor = 1800



    def get(self):
        """Return Nx6 DataFrame of all line annotations
        """
        with self.viewer.txn() as state:
            lines = state.layers["correspondences"].layer.annotations
            data = np.array([list(l.point_a) + list(l.point_b)
                             for l in lines if l.type=='line'])
            if len(data) > 0:
                return pd.DataFrame(data=data, columns=['x0','y0','z0','x1','y1','z1'])
            return None

    def set(self, data):
        """Set line annotations from DataFrame of points x0,y0,z0,x1,y1,z1
        """
        with self.viewer.txn() as state:
            lines = [neuroglancer.LineAnnotation(pointA=row[['x0','y0','z0']].to_list(), 
                                                 pointB=row[['x1','y1','z1']].to_list(), 
                                                 id=str(k)) for k, row in data.iterrows()] 
            state.layers["correspondences"].layer.annotations = lines

    def clear(self):
        """Remove all annotations
        """
        with self.viewer.txn() as state:
            state.layers["correspondences"].layer.annotations = [] 

    def load(self, path):
        try:
            lines = pd.read_csv(path, index_col=0)
            self.set(lines)
            print('Loaded annotations from {}'.format(path))
        except:
            print('Error: {}'.format(sys.exc_info()[0]))
            import pdb; pdb.set_trace()
            pass

    def save(self, path):
        try:
            lines = self.get()
            if lines is not None:
                lines.to_csv(path, header=True, index=True)
                print('Saved annotations to {}'.format(path))
        except:
            print('Error: {}'.format(sys.exc_info()[0]))
            import pdb; pdb.set_trace()
            pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-a',
        '--bind-address',
        help='Bind address for Python web server.  Use 127.0.0.1 '
        '(the default) to restrict access to browers running on the local machine, '
        'use 0.0.0.0 to permit access from remote browsers.')
    parser.add_argument(
        '--static-content-url', 
        help='Obtain the Neuroglancer client code from the specified URL.',
        # TODO: merge in neuroglancer==2.0.0 changes from Jeremy to seunglab fork
        default='https://neuromancer-seung-import.appspot.com/')
    parser.add_argument(
        '-s',
        '--source',
        help='path to datasource')
    parser.add_argument(
        '-c',
        '--coords',
        nargs=3,
        type=int,
        help='starting coordinates')
    args = parser.parse_args()
    c = CorrespondencesController(**vars(args))
    print(c.viewer)
    # TODO: webbrowser doesn't work on seungworkstation12 with Chrome with 
    # neuroglancer URLs, but does for others
    # webbrowser.open_new(c.viewer.get_viewer_url())
    cmd = None
    while cmd != 'q':
        cmd = input('Enter command (l: load, s: save, x: clear, q: quit): ')
        if cmd == 'l':
            path = input('Enter filename to load: ')
            c.load(path)
        elif cmd == 's':
            path = input('Enter path to save annotations: ')
            c.save(path)
        elif cmd == 'x':
            c.clear()