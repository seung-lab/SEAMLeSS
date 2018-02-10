
import numpy as np
import cavelab as cl

# Replace hyperparams
test_data = np.load('data/evaluate/simple_test.npy')
model = torch.load('logs/neuroflow/model.pt')

# given input x1, x2 compute flow
def compute_flow(x1,x2):
    o = np.zeros(x1.shape)
    return o

# given x1 and optical flow apply
def render(x, o):
