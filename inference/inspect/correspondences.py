import neuroglancer
from cloudvolume.lib import Bbox, Vec
import numpy as np

class Controller:
    """Simple class to set & get point-pairs from neuroglancer
    """

    def __init__(self, base_url='https://neuromancer-seung-import.appspot.com/', layer=''):
        neuroglancer.set_static_content_source(url=base_url)
        self.viewer = neuroglancer.Viewer()
        self.set_layout()
        self.layer = layer

    def set_layout(self, s='xy'):
        with self.viewer.txn() as state:
            state.layout = neuroglancer.DataPanelLayout(s)

    def set_layer(self, layer):
        with self.viewer.txn() as state:
            state.layers[layer['name']] = neuroglancer.ImageLayer(source=layer['url'])

    def set(self, points):
        with self.viewer.txn() as state:
            state.layers["correspondences"] = neuroglancer.AnnotationLayer()
            lines = [neuroglancer.LineAnnotation(pointA=pts[0], pointB=pts[1], 
                                     id=str(k)) for k, pts in enumerate(points)]
            state.layers["correspondences"].layer.annotations = lines

    def set_z(self, z):
        with self.viewer.txn() as state:
            state.voxel_coordinates = z

    def get_position(self):
        return self.viewer.state.voxel_coordinates

    def get_zoom(self):
        return self.viewer.state.navigation.zoom_factor

    def get_bbox(self):
        """Return square 2D bbox at approximate center of viewport
        """
        z2p = 64 # zoomFactor to bbox radius in pixels @ MIP0
        pos = Vec(*self.get_position())
        zoom = self.get_zoom()
        return Bbox(pos-Vec(z2p*zoom, z2p*zoom, 0), 
                    pos+Vec(z2p*zoom, z2p*zoom, 1))

if __name__ == '__main__':
    c = Controller()

