#import cavelab
import numpy as np
from os import listdir
import cavelab as cl

direct = '/projects/CrackFoldDetector/data/images/pinky100_v1/'
def find_filenames(path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

points = find_filenames(direct, '.npy')
# 1231 Crac
# 23 thin fold
# 93 border

class cf_data(object):
    """docstring for cf_data."""
    def __init__(self, src, mip=1):
        super(cf_data, self).__init__()
        self.pinky_prealigned = cl.Cloud(src, mip=mip)
        self.mip = mip

    def get_sample(self, x_y_z, size=256):
        (x,y,z) = x_y_z
        oset = size*(2**(self.mip-1))
        _image = self.pinky_prealigned.read_global((x-oset,y-oset,z),(size,size, 1))
        _template = self.pinky_prealigned.read_global((x-oset,y-oset,z+1),(size,size, 1))
        return np.squeeze(_image), np.squeeze(_template)

#[60905 52196    19]
src = 'gs://neuroglancer/pinky100_v0/image_single_slices/'
data = cf_data(src, mip=4)
#117564, 41726, 1987 - fold
#60905, 52196, 19 - crack

im, tmp = data.get_sample((117488, 41500, 1987), size=512)
print(im)
print(im.shape)
cl.visual.save(im/256.0, 'dump/image_fold')
cl.visual.save(tmp/256.0, 'dump/template_fold')
np.save('data/trivial/image_fold', im/256.0)
np.save('data/trivial/template_fold', tmp/256.0)
