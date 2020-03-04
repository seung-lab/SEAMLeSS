import os
import numpy as np
from boundingbox import BoundingBox, BoundingCube
from mipless_cloudvolume import MiplessCloudVolume
from fields import FieldCloudVolume 
from cloudsample import cloudsample_multicompose

class PairwiseFields():

    def __init__(self, path, offsets, bbox, mip, pad, **kwargs):
        """Manage set of FieldCloudVolumes that contain fields between neighbors
        
        We use the following notation to define a pairwise field which aligns
        section z to section z+k.
        
            $f_{z+k \leftarrow z}$
        
        It's easiest to interpret this as the displacement field which warps
        section z to look like section z+k. 

        We define the offset as the distance between the source and the target. So
        in the case above, the offset is k.

        We store this field in a CloudVolume where the path indicates the offset,
        and the actual field will be stored at cv[..., z].
        
        One purpose of this class is to easily compose pairwise fields together.
        For example, if we wanted to create the field:
        
            $f_{z+k \leftarrow z_j} \circ f_{z+j \leftarrow z}$ 
        
        Then we can access it with the convention:

            ```
            F = PairwiseFields(path, offsets, bbox, mip)
            f = F[(z+k, z+j, z)]
            ```
        
        Args:
            path: str to directory with pairwise field format
                ./{OFFSETS}
                f_{z+offset \leftarrow z} is stored in OFFSET[Z]
            offsets: list of ints indicating offset (the distance from source to
                targets).
            bbox: BoundingBox
            mip: int for MIP level of fields
            pad: int for amount of padding to use in composing fields
            kwargs: will be passed to MIplessCloudVolume
        """
        self.offsets = offsets
        self.bbox = bbox
        self.mip = mip
        self.pad = pad
        self.cvs = {}
        for o in offsets:
            cv_path = os.path.join(path, str(o))
            self.cvs[o] = MiplessCloudVolume(cv_path, obj=FieldCloudVolume, **kwargs)

    def __setitem__(self, tgt_to_src, field):
        """Save pairwise field at ./{OFFSET}[:,:,z]
        """
        if len(tgt_to_src) != 2:
            raise ValueError('len(tgt_to_src) is {} != 2. '
                             'Pairwise fields are only defined between '
                             'a pair of sections.'.format(len(tgt_to_src)))
        tgt, src = tgt_to_src
        offset = tgt - src
        cv = self.cvs[offset][self.mip]
        bcube = BoundingCube.from_bbox(self.bbox, zs=src)
        cv[bcube] = field

    def __getitem__(self, tgt_to_src):
        """Get field created by composing fields accessed by z_list[::-1]

        Args:
            tgt_to_src: list of ints, sorted from target to source, e.g.
                f_{0 \leftarrow 2} \circ f_{2 \leftarrow 3} : [0, 2, 3]
        """
        if len(tgt_to_src) < 2:
            raise ValueError('len(tgt_to_src) is {} < 2. '
                             'Pairwise fields must have at least a pair '
                             'of sections to be accessed/created.'.format(len(tgt_to_src)))
        elif len(tgt_to_src) == 2:
            tgt, src = tgt_to_src
            offset = tgt - src
            if offset not in self.offsets:
                raise ValueError('Requested offset {} is unavailable and likely '
                                 'need to be computed.'.format(offset))
            cv = self.cvs[offset][self.mip]
            bcube = BoundingCube.from_bbox(self.bbox, zs=src)
            return cv[bcube]
        else:
            offsets = np.array([t-s for t,s in zip(tgt_to_src[:-1], tgt_to_src[1:])])
            unavailable = any([o not in self.offsets for o in offsets])
            if unavailable:
                raise ValueError('Requested offsets {} are unavailable and likely '
                                 'need to be computed.'.format(offsets[unavailable]))
            cvs = [self.cvs[o] for o in offsets]
            return cloudsample_multicompose(field_list=cvs, 
                                            z_list=tgt_to_src[1:], 
                                            bbox=self.bbox, 
                                            mip_list=[self.mip]*len(cvs),
                                            dst_mip=self.mip,
                                            factors=None,
                                            pad=self.pad)
        

