

import numpy as np
import draw
import cavelab as cl
import torch
from torch.autograd import Variable
from util import get_identity


model = torch.load('logs/MVP/model.pt')
model.cuda()

# given input x1, x2 compute flow
def compute_flow(image,target):
    #image = cl.image_processing.resize(image, (1/2.0, 1/2.0))
    #target = cl.image_processing.resize(target, (1/2.0, 1/2.0))

    image = np.tile(image[:256,:256],(8,1,1))
    target = np.tile(target[:256,:256],(8,1,1))
    image = Variable(torch.from_numpy(image).cuda(),  requires_grad=False)
    target = Variable(torch.from_numpy(target).cuda(),  requires_grad=False)

    x = torch.stack([image, target], dim=1)

    ys, Rs, rs = model(x)
    return ys[-1].data.cpu().numpy()[0], Rs[-1].data.cpu().numpy()[0]

### Load the data
test_data = np.load('data/evaluate/simple_test.npy').astype(np.float32)

### Input formatting
x1 = test_data[:,:,3]/256.0
x2 = test_data[:,:,4]/256.0

### Compute
y, R = compute_flow(x1, x2)

identity = get_identity(batch_size=1, width=R.shape[-1])
R = R - identity[0]
R = np.transpose(R, (1,2,0))

hsv, grid = cl.visual.flow(R)

#### Draw
cl.visual.save(y, 'dump/image')
cl.visual.save(hsv, 'dump/target')
cl.visual.save(grid, 'dump/pred')
