import neuroglancer
import webbrowser
import argparse
import numpy as np
import pandas as pd
import sys

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