class Mask:
    def __init__(self, cv_path=None, mip=None, val=None, op='eq', dtype='uint8',
            coarsen_count=0):
        self.cv_path = cv_path
        self.cv = None
        self.mip = mip
        self.val = val
        self.op = op
        self.dtype = dtype
        self.coarsen_count = coarsen_count
