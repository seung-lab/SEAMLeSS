from torch.autograd import Variable
from torch.utils.data import DataLoader
import sys
import os
from pyramid import PyramidTransformer
from stack_dataset import StackDataset
from aug import aug_stacks
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--count', type=int, default=5)
parser.add_argument('--dataset', type=int, default=0)
parser.add_argument('--archive', type=str, default='pt/fprod_correct_enc6.pt')
parser.add_argument('--num_targets', type=int, default=1)
args = parser.parse_args()

if not os.path.isdir('inspect'):
    os.makedirs('inspect')
if not os.path.isdir('inspect/{}'.format(args.archive[3:-3])):
    os.makedirs('inspect/{}'.format(args.archive[3:-3]))

hm_dataset = StackDataset(os.path.expanduser('~/../eam6/basil_raw_cropped_train_mip5.h5'), None, None, basil=True)
lm_dataset1 = StackDataset(os.path.expanduser('~/../eam6/full_father_train_mip2.h5'), None, None, basil=True, lm=True) # dataset pulled from all of Basil
lm_dataset2 = StackDataset(os.path.expanduser('~/../eam6/dense_folds_train_mip2.h5'), None, None, basil=True, lm=True) # dataset focused on extreme folds
datasets = [hm_dataset, lm_dataset1, lm_dataset2]
train_dataset = datasets[args.dataset]
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=5, pin_memory=True)

model = PyramidTransformer.load(args.archive, height=7, dim=1280, skips=0, k=7, dilate=False, unet=False, num_targets=args.num_targets)

for t, tensor_dict in enumerate(train_loader):
    if t == args.count:
        break

    X, mask_stack = tensor_dict['X'], tensor_dict['m']
    # Get inputs
    X = Variable(X, requires_grad=False)
    mask_stack = Variable(mask_stack, requires_grad=False)
    X, mask_stack = X.cuda(), mask_stack.cuda()
    stacks, top, left = aug_stacks([X, mask_stack], padding=128)
    X, mask_stack = stacks[0], stacks[1]

    model.apply(X[0,0],X[0,1],vis='inspect/{}/sample{}'.format(args.archive[3:-3], t))
