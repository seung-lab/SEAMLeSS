import simplejson

class Mask:
    def __init__(self, cv_path=None, mip=None, val=None, op='eq', dtype='uint8',
            coarsen_count=0, mult=1.0):
        self.cv_path = cv_path
        self.cv = None
        self.mip = mip
        self.val = val
        self.op = op
        self.mult = mult
        self.dtype = dtype
        self.coarsen_count = coarsen_count

    @classmethod
    def from_json(cls, json):
        obj = cls(**json)
        return obj

    def __json__(self):
        return self.to_dict()

    to_json = __json__

    def to_dict(self):
        return {
            "cv_path": self.cv_path,
            "mip": self.mip,
            "val": self.val,
            "op": self.op,
            "mult": self.mult,
            "dtype": self.dtype,
            "coarsen_count": self.coarsen_count
        }
