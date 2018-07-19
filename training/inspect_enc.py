from torch.autograd import Variable
from torch.utils.data import DataLoader
import sys
import os
from pyramid import PyramidTransformer
from stack_dataset import StackDataset
from aug import aug_stacks, aug_input
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--count', type=int, default=5)
parser.add_argument('--dataset', type=int, default=0)
parser.add_argument('--archive', type=str, default='pt/fprod_correct_enc6.pt')
args = parser.parse_args()

if not os.path.isdir('inspect'):
    os.makedirs('inspect')
if not os.path.isdir('inspect/{}'.format(args.archive[3:-3])):
    os.makedirs('inspect/{}'.format(args.archive[3:-3]))

hm_dataset = StackDataset(os.path.expanduser('~/../eam6/mip5_mixed.h5'), mip=5)
datasets = [hm_dataset]
train_dataset = datasets[args.dataset]
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=5, pin_memory=True)

model = PyramidTransformer.load(args.archive, height=8, dim=1152, skips=0, k=7)

for t, tensor_dict in enumerate(train_loader):
    if t == args.count:
        break

    X = tensor_dict['X']
    # Get inputs
    X = Variable(X, requires_grad=False).cuda()
    stacks, top, left = aug_stacks([X], padding=0)
    X = stacks[0]
    src, target = X[0,0].clone(), X[0,1].clone()

    model.apply(aug_input(src)[0],aug_input(target)[0],vis='inspect/{}/sample{}'.format(args.archive[3:-3], t))
