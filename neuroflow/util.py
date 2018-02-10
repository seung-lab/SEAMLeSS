import numpy as np
import cavelab as cl



def get_identity(batch_size=8, width=256):
    identity = np.zeros((batch_size,2,width,width), dtype=np.float32)
    identity[:,0,:,:] = np.arange(width)/(width/2)-1
    identity[:,1,:,:] = np.transpose(identity, axes = [0,1,3,2])[:,0,:,:]
    return identity
