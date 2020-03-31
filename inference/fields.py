import math
import torch
import torchfields
import numpy as np
from copy import deepcopy

class Field():
    """Wrapper to maintain Field with absolute displacements at MIP0

    This class prevents need to keep track of relative displacements.
    """
    def __init__(self, data, bbox, *args, **kwargs):
        """Fields in Eulerian (pull) format with absolute displacements at MIP0

        Args:
            data: sequence of data 
            bbox: BoundingBox of field 
        """
        self.field = None
        if isinstance(data, torch.Field):
            self.field = data
        elif data is not None:
            self.field = torch.Field(data, *args, **kwargs)
        self.bbox = bbox
        self.rel = False 

    @classmethod
    def from_torchfield(cls, field, bbox):
        return cls(data=field, bbox=bbox)

    def new(self, field):
        """Create new field with same bbox 
        """
        return Field.from_torchfield(field=field, bbox=self.bbox)

    def copy(self):
        return deepcopy(self)


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
    def shape(self):
        """Size of bbox at MIP0
        """
        return self.field.shape

    @property
    def device(self):
        return self.field.device

    @property
    def is_cuda(self):
        return self.field.is_cuda

    @property
    def mip(self):
        """Current MIP level of field
        """
        return int(math.log2(self.bbox.size[0] / self.field.shape[-2]))

    def cpu(self):
        return self.field.cpu()

    def numpy(self):
        return self.field.data.numpy()

    def to(self, device):
        return self.new(self.field.to(device=device))

    def profile(self, **kwargs):
        return self.field.mean_finite_vector(**kwargs)

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
        g.rel = True
        g.to_abs()
        self.to_abs()
        x.to_abs()
        return g

    # TODO
    # def __getitem__(self, slices):
    #     x = self.field[slices]

    def equal_field(self, x):
        if isinstance(x, Field):
            return torch.equal(self.field, x.field) 
        return False

    def allclose(self, x, **kwargs):
        if isinstance(x, Field):
            return torch.allclose(self.field, x.field, **kwargs) and (self.bbox == x.bbox)
        return False
        
    def __eq__(self, x):
        return self.equal_field(x) and (self.bbox == x.bbox)

    def __mul__(self, c):
        x = self.field * c 
        return self.new(x)

    def __neg__(self):
        return self.__mul__(-1)

    def __div__(self, c):
        return self.__mul__(1/c)

    def __add__(self, c):
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


