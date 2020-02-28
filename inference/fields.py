from cloudvolume import CloudVolume 
import torch
import torchfields

class Field(torchfields.DisplacementField):
    """Wrapper to maintain Field with absolute displacements at MIP0

    This class prevents need to keep track of relative displacements.
    """

    def __init__(self, bbox, **kwargs):
        """Fields in Eulerian (pull) format with absolute displacements at MIP0

        Args:
            bbox: BoundingBox of field 
        """
        super().__init__(**kwargs)
        self.bbox = bbox
        self.rel = False 

    @property
    def size(self):
        """Size of bbox at MIP0
        """
        return self.bbox.size(mip=0)

    @property
    def mip(self):
        """Current MIP level of field
        """
        return int(math.log2(self.bbox.size(mip=0)[0] / self.shape[0]))

    def to_rel(self):
        if not self.rel:
            self.from_pixels(size=self.size)
            self.rel = True

    def to_abs(self):
        if self.rel:
            self.pixels(size=self.size)

    def __call__(self, x, **kwargs):
        self.to_rel()
        if isinstance(x, Field):
            x.to_abs()
        super().__call__(x, **kwargs)
        self.to_abs()


class FieldCloudVolume(CloudVolume):

    def __init__(self, as_int16=False, device='cpu', **kwargs):
        self.as_int16 = as_int16
        self.device = device
        super().__init__(**kwargs)

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

