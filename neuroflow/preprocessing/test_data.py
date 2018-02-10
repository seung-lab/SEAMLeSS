#Implement testing data
#1500-1510

# 56363, 36229, 1503 - 1508

import numpy as np
import cavelab as cl

(x,y,z) = (55289, 36000, 1503)
size = 512
oset = 0

#Load hyperparams
hparams = cl.hparams(name="preprocessing")
cloud = cl.Cloud(hparams.cloud_src, mip=hparams.cloud_mip, cache=False, bounded = False, fill_missing=True)
eval_data = cloud.read_global((x-oset,y-oset,z),(size,size, 8))

np.save('data/evaluate/simple_test', eval_data)

cl.visual.save(eval_data[:,:,3]/256.0, 'dump/image_test')
cl.visual.save(eval_data[:,:,4]/256.0, 'dump/image_test_1')
print(eval_data.shape)
