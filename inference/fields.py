import math
import torch
import torchfields
from copy import deepcopy
from cloudvolume import CloudVolume 
from mipless_cloudvolume import MiplessCloudVolume

class Field():
    """Wrapper to maintain Field with absolute displacements at MIP0

    This class prevents need to keep track of relative displacements.
    """
    def __init__(self, data, bbox, *args, **kwargs):
        """Fields in Eulerian (pull) format with absolute displacements at MIP0

        Args:
            data: sequence of data (
            bbox: BoundingBox of field 
        """
        self.field = torch.Field(data, *args, **kwargs)
        self.bbox = bbox
        self.rel = False 

    def __repr__(self):
        return 'data:\t{}\n' \
               'bbox:\t{}\n' \
               'mip:\t{}'.format(self.field, self.bbox, self.mip)

    @property
    def size(self):
        """Size of bbox at MIP0
        """
        return self.bbox.size

    @property
    def mip(self):
        """Current MIP level of field
        """
        return int(math.log2(self.bbox.size[0] / self.field.shape[-2]))

    def new(self, field):
        """Create new field with same bbox 
        """
        x = self.copy()
        x.field = field
        return x

    def copy(self):
        return deepcopy(self)

    def to_rel(self):
        if not self.rel:
            self.field = self.field.from_pixels(size=self.size)
            self.rel = True

    def to_abs(self):
        if self.rel:
            self.field = self.field.pixels(size=self.size)
            self.rel = False

    def __call__(self, x, **kwargs):
        self.to_rel()
        if isinstance(x, Field):
            x.to_rel()
        g = self.field(x.field)
        g = self.new(g)
        g.to_abs()
        self.to_abs()
        x.to_abs()
        return g

    def equal_field(self, x):
        if isinstance(x, Field):
            return torch.equal(self.field, x.field) 
        return False
        
    def __eq__(self, x):
        return self.equal_field(x) and (self.bbox == x.bbox)

    def __mul__(self, c):
        assert(isinstance(c, (int, float, complex)))
        x = self.field * c 
        return self.new(x)

    def __neg__(self):
        return self.__mul__(-1)

    def __div__(self, c):
        return self.__mul__(1/c)

    def __add__(self, c):
        assert(isinstance(c, (int, float, complex)))
        x = self.field + c 
        return self.new(x)

    def __sub__(self, c):
        return self.__add__(-c)

    def up(self, *args, **kwargs):
        x = self.field.up(*args, **kwargs)
        return self.new(x)

    def down(self, *args, **kwargs):
        x = self.field.down(*args, **kwargs)
        return self.new(x)

    def is_identity(self, *args, **kwargs):
        return self.field.is_identity(*args, **kwargs)


class FieldCloudVolume(CloudVolume):

    def __new__(cls, *args, **kwargs):
        as_int16 = kwargs.pop('as_int16')
        device = kwargs.pop('device')
        cv = CloudVolume.__new__(cls, *args, **kwargs) 
        kwargs['as_int16'] = as_int16
        kwargs['device'] = device
        return cv

    def __init__(self, *args, **kwargs):
        self.as_int16 = kwargs.pop('as_int16')
        self.device = kwargs.pop('device')
        super().__init__(*args, **kwargs)

    def __getitem__(self, slices):
        """Get field cutout as absolute residuals

        Args:
            slices:

        Returns:
            TorchField with absolute residuals
        """
        field = super().__getitem__(slices)
        field = np.transpose(field, (2,0,1,3))
        if self.as_int16:
          field = np.float32(field) / 4
        return Field(data=field, device=self.device)

    def __setitem__(self, slices, field):
        """Save field (must be in absolute residuals)

        Args:
            slices: 
            field: TorchField with absolute residuals
        """
        field = field.tensor()
        if field.device.contains('gpu'):
            field = field.cpu()
        field = np.transpose(field.numpy(), (1,2,0,3))
        if self.as_int16:
            if(np.max(field) > 8192 or np.min(field) < -8191):
                print('Value in field is out of range of int16 ' \
                      'max: {}, min: {}'.format(np.max(field),
                                                np.min(field)), flush=True)
            field = np.int16(field * 4)
        super().__setitem__(slices, field)


class MiplessFieldCloudVolume(MiplessCloudVolume):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, obj=FieldCloudVolume)

def profile_field(field):
    return field.mean_finite_vector()

