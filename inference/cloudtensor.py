import json
import numpy as np
import torch

from fields import Field
from cloudvolume import CloudVolume

class CloudTensor():
    """Extend CloudVolume to retrieve/store torch.Tensors
    Retrieval and storage adopt mipless BoundingCubes
    """
    def __init__(self, *args, **kwargs):
        """Opt to maintain CloudVolume as attribute

        Args:
            device: str for torch.device
        """
        self.device = kwargs.pop('device')
        self.cv = CloudVolume(*args, **kwargs)

    @classmethod
    def from_cv(cls, cv, device):
        """Create CloudTensor from existing CloudVolume
        """
        obj = cls.__new__(cls)
        super(CloudTensor, obj).__init__()
        obj.device = device
        obj.cv = cv
        return obj

    @classmethod
    def create_new_info(cls, *args, **kwargs):
        return CloudVolume.create_new_info(*args, **kwargs)

    @property
    def dtype(self):
        return self.cv.dtype

    @property
    def info(self):
        return self.cv.info

    def commit_info(self):
        self.cv.commit_info()

    def commit_provenance(self):
        self.cv.commit_provenance()

    def __getitem__(self, bcube):
        """Get tensor from BoundingCube

        Args:
            bcube: BoundingCube (MiplessBoundingBox + z range)

        Returns:
            torch.Tensor with dtype=float32
        """
        slices = bcube.to_slices(mip=self.cv.mip)
        img = self.cv[slices]
        img = np.transpose(img, (2,3,0,1))
        if self.dtype == 'uint8':
            img = np.divide(img, float(255.0), dtype=np.float32)
        img = torch.from_numpy(img)
        return img.to(device=self.device)

    def __setitem__(self, bcube, img):
        """Save image

        Args:
            bcube: BoundingCube (MiplessBoundingBox + z range)
            img: image object
        """
        if img.is_cuda:
            img = img.cpu()
        slices = bcube.to_slices(mip=self.cv.mip)
        patch = np.transpose(img.numpy(), (2,3,0,1))
        if self.dtype == 'uint8':
            patch = (np.multiply(patch, 255)).astype(np.uint8)
        self.cv[slices] = patch

class CloudField(CloudTensor):
    """Extend CloudVolume to retrieve/store torchfields.DisplacementFields
    """
    def __getitem__(self, bcube):
        """Get field cutout as absolute residuals

        Args:
            bcube: BoundingCube (MiplessBoundingBox + z range)

        Returns:
            Field object (with absolute residuals)
        """
        slices = bcube.to_slices(mip=self.cv.mip)
        field = self.cv[slices]
        field = np.transpose(field, (2,3,0,1))
        if self.cv.dtype == 'int16':
          field = np.float32(field) / 4
        return Field(data=field, bbox=bcube.bbox, device=self.device) 

    def __setitem__(self, bcube, field):
        """Save field (must be in absolute residuals)

        Args:
            bcube: BoundingCube (MiplessBoundingBox + z range)
            field: Field object
        """
        if field.is_cuda:
            field = field.cpu()
        field = np.transpose(field.numpy(), (2,3,0,1))
        if self.cv.dtype == 'int16':
            if(np.max(field) > 8192 or np.min(field) < -8191):
                print('Value in field is out of range of int16 ' \
                      'max: {}, min: {}'.format(np.max(field),
                                                np.min(field)), flush=True)
            field = np.int16(field * 4)
        slices = bcube.to_slices(mip=self.cv.mip)
        self.cv[slices] = field


class MiplessCloudTensor():
    """Multi-MIP access to CloudTensors using the same path
    """
    def __init__(self, path, device='cpu', **kwargs):
        self.path = path
        self.device = device
        kwargs['device'] = device
        self.kwargs = kwargs
        self.cvs = {}

    @property
    def cloudtype(self):
        return CloudTensor

    def mkdir(self):
        cv = self.cloudtype(self.path, **self.kwargs)
        cv.commit_info()
        cv.commit_provenance()

    def exists(self):
        raise NotImplementedError

    def serialize(self):
        contents = {
            "path" : self.path,
            "device" : self.device,
        }
        s = json.dumps(contents)
        return s

    @classmethod
    def deserialize(cls, s, cache={}, **kwargs):
        kwargs['bounded'] = False
        kwargs['progress'] = False
        kwargs['autocrop'] = False
        kwargs['non_aligned_writes'] = False
        kwargs['cdn_cache'] = False
        kwargs['fill_missing'] = True 
        if s in cache:
            return cache[s]
        else:
            mcv = cls(s, **kwargs)
            cache[s] = mcv
            return mcv

    def info(self, mip=0):
        """Retrieve the info file
        """
        if len(self.cvs) > 0:
            return self.cvs.get(list(self.cvs.keys())[0]).info
        info = self.kwargs.get('info')
        if info is None:
            cv = self.cloudtype(self.path, mip=mip, **self.kwargs)
            info = cv.info
        return info

    def create(self, mip):
        print('Creating CloudVolume for {0} at MIP{1}'.format(self.path, mip))
        self.cvs[mip] = self.cloudtype(self.path, mip=mip, **self.kwargs)

    def __getitem__(self, mip):
        if mip not in self.cvs:
            self.create(mip)
        return self.cvs[mip]
 
    def __repr__(self):
        return self.path

class MiplessCloudField(MiplessCloudTensor):

    @property
    def cloudtype(self):
        return CloudField 
