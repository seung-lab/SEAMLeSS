"""Train a network.

This is the main module which is invoked to train a network.

Running:
    To begin training, run this on a machine with a GPU.
    (Don't forget to install the required dependencies in requirements.txt)

        $ python3 train.py [--param1_name VALUE1 --param2_name VALUE2 ...]
            EXPERIMENT_NAME

Example:
        $ python3 train.py --state_archive pt/SOME_ARCHIVE.pt --size 8
            --lambda1 2 --lambda2 0.04 --lambda3 0 --lambda4 5 --lambda5 0
            --mask_smooth_radius 75 --mask_neighborhood_radius 75 --lr 0.0003
            --trunc 0 --fine_tuning --hm --padding 0 --vis_interval 5
            --lambda6 1 fine_tune_example

Todo:
    * Refactor main method
    * Expand docstring

"""

import os
from os.path import expanduser
import time
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, ConcatDataset
from torch.nn.parallel import data_parallel
from torchvision import transforms
import argparse

import masks
from stack_dataset import compile_dataset, Normalize, ToFloatTensor, RandomRotateAndScale 
from pyramid import PyramidTransformer
from helpers import reverse_dim, downsample
from aug import aug_stacks, aug_input, rotate_and_scale, crack, displace_slice
from vis import visualize_outputs
from loss import similarity_score, smoothness_penalty

class ModelWrapper(nn.Module):
    """Abstraction to allow for multi-GPU training
    """
    def __init__(self, args, model):
        super(ModelWrapper, self).__init__()
        self.args = args
        self.model = model
        self.trunclayer = 0 
        self.smooth_factor = 1
        self.set_trunclayer(self.trunclayer)
        self.update_smooth_factor()
        self.padding = args.padding
        self.dim = args.dim + self.padding
        self.similarity = similarity_score(should_reduce=False)
        self.smoothness = smoothness_penalty(args.penalty)
        # self.norm = Normalizer(5 if args.hm else 2)

    def set_trunclayer(self, trunc):
        self.trunclayer = trunc

    def update_smooth_factor(self):
        self.smooth_factor = 1 if self.trunclayer == 0 and self.args.fine_tuning else 0.05

    def save(self, name):
        torch.save(self.model.state_dict(), 'pt/' + name + '.pt')

    def run_pair(self, src, tgt, src_mask=None, tgt_mask=None, train=True):
        input_src = src.clone()
        input_tgt = tgt.clone()

        src_cutout_masks = []
        tgt_cutout_masks = []

        # if train and not self.args.skip_sample_aug:
        #     input_src, src_cutout_masks = aug_input(input_src)
        #     input_tgt, tgt_cutout_masks = aug_input(input_tgt)
        # elif train:
        #     print('Skipping sample-wise augmentation.')

        # src = Variable(torch.FloatTensor(self.norm.apply(
        #     src.data.cpu().numpy()))).cuda()
        # tgt = Variable(torch.FloatTensor(self.norm.apply(
        #     tgt.data.cpu().numpy()))).cuda()
        # input_src = Variable(torch.FloatTensor(self.norm.apply(
        #     input_src.data.cpu().numpy()))).cuda()
        # input_tgt = Variable(torch.FloatTensor(self.norm.apply(
        #     input_tgt.data.cpu().numpy()))).cuda()

        rf = self.run_sample(
            src, tgt, input_src, input_tgt, src_mask, tgt_mask,
            src_cutout_masks, tgt_cutout_masks, train)
        if rf is not None:
            if not self.args.pe_only:
                cost = (rf['similarity_error']
                        + self.args.lambda1
                        * self.smooth_factor * rf['smoothness_error'])
                (cost/2).backward(retain_graph=self.args.lambda6 > 0)
            else:
                (rf['contrast_error']/2).backward()

        src = reverse_dim(reverse_dim(src, 0), 1)
        tgt = reverse_dim(reverse_dim(tgt, 0), 1)
        input_src = reverse_dim(reverse_dim(input_src, 0), 1)
        input_tgt = reverse_dim(reverse_dim(input_tgt, 0), 1)
        if src_mask is not None:
            src_mask = reverse_dim(reverse_dim(src_mask, 0), 1)
        if tgt_mask is not None:
            tgt_mask = reverse_dim(reverse_dim(tgt_mask, 0), 1)
        src_cutout_masks = [reverse_dim(reverse_dim(m, -2), -1)
                            for m in src_cutout_masks]
        tgt_cutout_masks = [reverse_dim(reverse_dim(m, -2), -1)
                               for m in tgt_cutout_masks]

        rb = self.run_sample(
            src, tgt, input_src, input_tgt, src_mask, tgt_mask,
            src_cutout_masks, tgt_cutout_masks, train)
        if rb is not None:
            if not self.args.pe_only:
                cost = (rb['similarity_error']
                        + self.args.lambda1
                        * self.smooth_factor * rb['smoothness_error'])
                (cost/2).backward(retain_graph=self.args.lambda6 > 0)
            else:
                (rb['contrast_error']/2).backward()

        ##################################
        # CONSENSUS PENALTY COMPUTATION ##
        ##################################
        if (not self.args.pe_only and self.args.lambda6 > 0 and rf is not None
                and rb is not None):
            smaller_dim = self.dim // (2**self.trunclayer)
            zmf = (rf['pred'] != 0).float() * (rf['input_tgt'] != 0).float()
            zmf = (zmf.detach().unsqueeze(0).view(1, 1, smaller_dim, smaller_dim)
                   .repeat(1, 2, 1, 1).permute(0, 2, 3, 1))
            zmb = (rb['pred'] != 0).float() * (rb['input_tgt'] != 0).float()
            zmb = (zmb.detach().unsqueeze(0).view(1, 1, smaller_dim, smaller_dim)
                   .repeat(1, 2, 1, 1).permute(0, 2, 3, 1))
            rffield, rbfield = rf['rfield'] * zmf, rb['rfield'] * zmb
            rbfield_reversed = -reverse_dim(reverse_dim(rbfield, 1), 2)
            consensus_diff = rffield - rbfield_reversed
            consensus_error_field = (consensus_diff[:, :, :, 0] ** 2
                                     + consensus_diff[:, :, :, 1] ** 2)
            mean_consensus = torch.mean(consensus_error_field)
            rf['consensus_field'] = rffield
            rb['consensus_field'] = rbfield_reversed
            rf['consensus_error_field'] = consensus_error_field
            rb['consensus_error_field'] = consensus_error_field
            rf['consensus'] = mean_consensus
            rb['consensus'] = mean_consensus
            consensus = self.args.lambda6 * mean_consensus
            consensus.backward()

        # print('type(rf), type(rb): {0} {1}'.format(type(rf), type(rb)))
        return rf, rb

    def run_sample(self, src, tgt, input_src, input_tgt, mask=None,
                   tgt_mask=None, src_cutout_masks=[],
                   tgt_cutout_masks=[], train=True):
        self.model.train(train)
        if self.args.dry_run:
            print('Bailing for dry run [not an error].')
            return None

        # if we're just training the pre-encoder, no need to do everything
        if self.args.pe_only:
            contrasted_stack = self.model.apply(input_src, input_tgt, 
                                           use_preencoder=self.args.pe)
            pred = contrasted_stack.squeeze()
            y = torch.cat((src.unsqueeze(0), tgt.squeeze().unsqueeze(0)), 0)
            contrast_err = self.similarity(pred, y)
            contrast_err_src = contrast_err[0:1]
            contrast_err_tgt = contrast_err[1:2]
            for m in src_cutout_masks:
                contrast_err_src = (contrast_err_src
                                    * m.view(contrast_err_src.size()).float())
            for m in tgt_cutout_masks:
                contrast_err_tgt = (contrast_err_tgt
                                       * m.view(contrast_err_tgt.size())
                                       .float())
            contrast_err = torch.mean(torch.cat((contrast_err_src,
                                                 contrast_err_tgt), 0))
            return {'contrast_error': contrast_err}

        # print('input_src/tgt device: {0} {1}'.format(input_src.device, input_tgt.device))
        pred, field, residuals = self.model.apply(input_src, input_tgt,
                                             self.trunclayer, use_preencoder=self.args.pe)
        # resample tensors that are in our source coordinate space with our
        # new field prediction to move them into tgt coordinate space so
        # we can compare things fairly
        src = downsample(self.trunclayer)(src.unsqueeze(0).unsqueeze(0))
        tgt = downsample(self.trunclayer)(tgt.unsqueeze(0).unsqueeze(0))
        pred = F.grid_sample(src, field, mode='bilinear')
        raw_mask = None
        if mask is not None:
            mask = downsample(self.trunclayer)(mask.unsqueeze(0).unsqueeze(0))
            raw_mask = mask
            mask = torch.ceil(F.grid_sample(mask, field, mode='bilinear', padding_mode='border'))
        if tgt_mask is not None:
            tgt_mask = downsample(self.trunclayer)(tgt_mask.unsqueeze(0).unsqueeze(0))
            tgt_mask = masks.dilate(tgt_mask, 1, binary=False)
        if len(src_cutout_masks) > 0:
            src_cutout_masks = [
                torch.ceil(F.grid_sample(m.float(), field, mode='bilinear', padding_mode='border')).byte()
                for m in src_cutout_masks]

        if len(tgt_cutout_masks) > 0:
             tgt_cutout_masks = [torch.ceil(downsample(self.trunclayer)(m.float())).byte() for m in tgt_cutout_masks]

        # first we'll build a binary mask to completely ignore
        # 'irrelevant' pixels
        similarity_binary_masks = []

        # mask away similarity error from pixels outside the boundary of
        # either prediction or tgt
        # in other words, we need valid data in both the tgt and prediction
        # to count the error from that pixel
        border_mask = masks.low_pass(masks.intersect([pred != 0, tgt != 0]))
        border_mask, no_valid_pixels = masks.contract(border_mask, 5,
                                                      return_sum=True)
        if no_valid_pixels:
            print('Skipping empty border mask.')
#             visualize_outputs(prefix('empty_border_mask') + '{}',
#                               {'src': input_src, 'tgt': input_tgt}
            return None
        similarity_binary_masks.append(border_mask)

        # mask away similarity error from pixels within cutouts in either the
        # source or tgt
        cutout_similarity_masks = src_cutout_masks + tgt_cutout_masks
        if len(cutout_similarity_masks) > 0:
            cutout_similarity_mask = masks.union(cutout_similarity_masks)
            cutout_similarity_mask = masks.invert(cutout_similarity_mask)
            similarity_binary_masks.append(cutout_similarity_mask)

        # mask away similarity error for pixels within a defect in the
        # tgt image
        if tgt_mask is not None:
            tgt_defect_similarity_mask = masks.contract(
                tgt_mask < 2, 2).detach()
            similarity_binary_masks.append(tgt_defect_similarity_mask)

        similarity_binary_mask = masks.intersect(similarity_binary_masks)

        # reweight remaining pixels for focus areas
        similarity_weights = Variable(torch.ones(border_mask.size())).cuda()
        if mask is not None:
            similarity_weights.data[
                masks.intersect([mask > 0, mask <= 1]).data
            ] = self.args.lambda4
            similarity_weights.data[
                masks.dilate(mask > 1, 2).data
            ] = self.args.lambda5

        similarity_weights *= similarity_binary_mask.float()

        # if torch.sum(similarity_weights[border_mask.data]).data[0] < self.args.eps:
        #     print('Skipping all zero similarity weights (factor == 0).')
#       #       visualize_outputs(prefix('zero_similarity_weights') + '{}',
#       #                         {'src': input_src, 'tgt': input_tgt})
        #     return None

        # similarity_mask_factor = (
        #     torch.sum(border_mask.float())
        #     / torch.sum(similarity_weights[border_mask.data])).data[0]
        similarity_mask_factor = 1

        # compute raw similarity error and masked/weighted similarity error
        similarity_error_field = self.similarity(pred, tgt)
        weighted_similarity_error_field = (similarity_error_field
                                           * similarity_weights
                                           * similarity_mask_factor)

        # perform masking/weighting for smoothness penalty
        smoothness_binary_mask = (masks.intersect([src != 0, tgt != 0])
                                  .view(similarity_binary_mask.size()))
        smoothness_weights = Variable(
            torch.ones(smoothness_binary_mask.size())).cuda()
        if raw_mask is not None:
            # everywhere we have non-standard smoothness, slightly reduce
            # the smoothness penalty
            smoothness_weights[raw_mask > 0] = self.args.lambda2
            # on top of cracks and folds only, significantly reduce
            # the smoothness penalty
            smoothness_weights[raw_mask > 1] = self.args.lambda3
        smoothness_weights = masks.contract(
            smoothness_weights, self.args.mask_smooth_radius,
            ceil=False, binary=False)
        smoothness_weights *= smoothness_binary_mask.float().detach()
        smoothness_weights = F.avg_pool2d(
            smoothness_weights, 2*self.args.mask_smooth_radius+1, stride=1,
            padding=self.args.mask_smooth_radius
        )**.5
        if raw_mask is not None:
            # reset the most significant smoothness penalty relaxation
            smoothness_weights[raw_mask > 1] = self.args.lambda3
        smoothness_weights = F.avg_pool2d(smoothness_weights, 5,
                                          stride=1, padding=2)
        if (torch.sum(smoothness_weights[smoothness_binary_mask.byte().data])
                .data[0] < self.args.eps):
             print('Skipping all zero smoothness weights (factor == 0).')
#             visualize_outputs(prefix('zero_smoothness_weights') + '{}',
#                               {'src': input_src, 'tgt': input_tgt})
             return None

        # smoothness_mask_factor = (dim**2. / torch.sum(
        #     smoothness_weights[smoothness_binary_mask.byte().data]).data[0])
        smoothness_mask_factor = 1

        smoothness_weights /= smoothness_mask_factor
        smoothness_weights = F.grid_sample(smoothness_weights.detach(), field, mode='bilinear', padding_mode='border')
        if self.args.hm:
            smoothness_weights = smoothness_weights.detach()
        smoothness_weights = smoothness_weights * border_mask.float().detach()

        rfield = field - self.model.pyramid.get_identity_grid(field.size()[-2], src.device)
        weighted_smoothness_error_field = self.smoothness(
            [rfield], weights=smoothness_weights)

        if mask is not None:
            hpred = pred.clone().detach()
            hpred[(mask > 1).detach()] = torch.min(pred).data[0]
        else:
            hpred = pred

        return {
            'src': src,
            'tgt': tgt,
            'input_src': downsample(self.trunclayer)(input_src.unsqueeze(0).unsqueeze(0)),
            'input_tgt': downsample(self.trunclayer)(input_tgt.unsqueeze(0).unsqueeze(0)),
            'field': field,
            'rfield': rfield,
            'residuals': residuals,
            'pred': pred,
            'hpred': hpred,
            'src_mask': mask,
            'raw_src_mask': raw_mask,
            'tgt_mask': tgt_mask,
            'similarity_error':
                torch.mean(weighted_similarity_error_field[border_mask]),
            'smoothness_error':
                torch.sum(weighted_smoothness_error_field),
            'smoothness_weights': smoothness_weights,
            'similarity_weights': similarity_weights,
            'similarity_error_field': weighted_similarity_error_field,
            'smoothness_error_field': weighted_smoothness_error_field
        }

    def forward(self, sample):
        """Run single pair of slices
        """
        # src = Variable(sample['src'], requires_grad=False).cuda() 
        # tgt = Variable(sample['tgt'], requires_grad=False).cuda() 
        # src, tgt = sample['src'], sample['tgt']
        src, tgt = sample['src'].squeeze(), sample['tgt'].squeeze()
        # print('src/tgt device: {0} {1}'.format(src.device, tgt.device))
        return self.run_pair(src, tgt)

def main():
    """Start training a net."""
    args = parse_args()
    name = args.name
    trunclayer = args.trunc
    skiplayers = args.skip
    size = args.size
    fall_time = args.fall_time
    padding = args.padding
    dim = args.dim + padding
    kernel_size = args.k
    log_path = 'out/' + name + '/'
    log_file = log_path + name + '.log'
    lr = args.lr
    fine_tuning = args.fine_tuning
    epoch = args.epoch
    start_time = time.time()
    history = []

    # GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu_ids)

    if args.mm:
        paths = [args.hm_src, args.lm_src, args.vhm_src]
    else:
        if args.hm:
            paths = [args.hm_src, args.vhm_src]
        else:
            paths = [args.lm_src]
    h5_paths = [expanduser(x) for x in paths]
    transform = transforms.Compose([Normalize(2), ToFloatTensor(), 
                                     RandomRotateAndScale()])
    train_dataset = compile_dataset(h5_paths, transform)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, pin_memory=True)

    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    if not os.path.isdir('pt'):
        os.makedirs('pt')

    with open(log_path + 'args.txt', 'a') as f:
        f.write(str(args))

    if args.state_archive is None:
        model = PyramidTransformer(
            size=size, dim=dim, skip=skiplayers, k=kernel_size).cuda()
    else:
        model = PyramidTransformer.load(
            args.state_archive, height=size, dim=dim, skips=skiplayers,
            k=kernel_size)
        for p in model.parameters():
            p.requires_grad = True
        model.train().cuda()

    model_wrapper = ModelWrapper(args, model)

    def opt(layer):
        params = []
        if not args.pe_only and not args.enc_only:
            if layer >= skiplayers and not fine_tuning:
                print('Training only layer {}'.format(layer))
                params.extend(model.pyramid.mlist[layer].parameters())
            else:
                print('Training all residual networks.')
                params.extend(model.pyramid.mlist.parameters())

            if fine_tuning or epoch < fall_time - 1:
                print('Training all encoder parameters.')
                params.extend(model.pyramid.enclist.parameters())
                params.extend(model.pyramid.pe.parameters())
            else:
                print('Freezing encoder parameters.')
        elif args.pe_only:
            print('Training pre-encoder only.')
            params.extend(model.pyramid.pe.parameters())
        elif args.enc_only:
            print('Training all encoder parameters only.')
            params.extend(model.pyramid.enclist.parameters())
            params.extend(model.pyramid.pe.parameters())

        lr_ = lr if not fine_tuning else lr * args.fine_tuning_lr_factor
        print(
            'Building optimizer for layer {} (fine tuning: {}, lr: {})'
            .format(layer, fine_tuning, lr_))
        return torch.optim.Adam(
            params, lr=lr_, weight_decay=0.0001 if args.pe_only else 0)

    def prefix(tag=None):
        if tag is None:
            return '{}{}_e{}_t{}_'.format(log_path, name, epoch, t)
        else:
            return '{}{}_e{}_t{}_{}_'.format(log_path, name, epoch, t, tag)

    print('=========== BEGIN TRAIN LOOP ============')
    for epoch in range(args.epoch, args.num_epochs):
        print('Beginning training epoch: {}'.format(epoch))

        errs = []
        penalties = []
        consensus_list = []
        contrast_errors = []
        smooth_factor = model_wrapper.smooth_factor

        if epoch % fall_time == 0 and (trunclayer > 0
                                       or args.trunc == 0):
            # only fine tune if running a tuning session
            fine_tuning = False or args.fine_tuning
            if epoch > 0 and trunclayer > 0:
                trunclayer -= 1
                model_wrapper.set_trunclayer(trunclayer)
            optimizer = opt(trunclayer)
        elif (epoch >= fall_time * size - 1
              or epoch % fall_time == fall_time - 1):
            if not fine_tuning:
                fine_tuning = True
                optimizer = opt(trunclayer)

        for t, sample in enumerate(train_loader):

            if len(args.gpu_ids) > 1:
                # print('using multiple gpus')
                rf, rb = data_parallel(model_wrapper, sample)
            else:
                # print('not using multiple gpus')
                rf, rb = model_wrapper(sample)

            if rf is not None:
                if not args.pe_only:
                    errs.append(rf['similarity_error'].data[0])
                    penalties.append(rf['smoothness_error'].data[0])
                    if args.lambda6 > 0 and 'consensus' in rf:
                        consensus_list.append(rf['consensus'].data[0])
                else:
                    contrast_errors.append(rf['contrast_error'].data[0])

            if rb is not None:
                if not args.pe_only:
                    errs.append(rb['similarity_error'].data[0])
                    penalties.append(rb['smoothness_error'].data[0])
                    if args.lambda6 > 0 and 'consensus' in rb:
                        consensus_list.append(rb['consensus'].data[0])
                else:
                    contrast_errors.append(rb['contrast_error'].data[0])

#             if (sample_idx == 0 and t % args.vis_interval == 0
#                     and not args.pe_only):
#                 visualize_outputs(prefix('forward') + '{}', rf)
#                 visualize_outputs(prefix('backward') + '{}', rb)
            ##################################
            # print('optimizer.step()')
            optimizer.step()
            # model_wrapper.model.zero_grad()
            optimizer.zero_grad()


        if args.pe_only:
            mean_contrast = (
                (sum(contrast_errors) / len(contrast_errors))
                if len(contrast_errors) > 0 else 0)
            print('Mean contraster error: {}'.format(mean_contrast))
            model_wrapper.save(name)
        # Save some info
        if len(errs) > 0:
            mean_err_train = sum(errs) / len(errs)
            mean_penalty_train = sum(penalties) / len(penalties)
            mean_consensus = (
                (sum(consensus_list) / len(consensus_list))
                if len(consensus_list) > 0 else 0)
            print(epoch, smooth_factor, trunclayer,
                  mean_err_train + args.lambda1
                  * mean_penalty_train * smooth_factor,
                  mean_err_train, mean_penalty_train, mean_consensus)
            history.append((
                time.time() - start_time,
                mean_err_train + mean_penalty_train * smooth_factor,
                mean_err_train, mean_penalty_train, mean_consensus))
            model_wrapper.save(name)
  
            print('Writing status to: {}'.format(log_file))
            with open(log_file, 'a') as log:
                for tr in history:
                    for val in tr:
                        log.write(str(val) + ', ')
                    log.write('\n')
                history = []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument(
        '--dry_run',
        help='tests data loading when passed, skipping training',
        action='store_true')
    parser.add_argument(
        '--vis_interval',
        help='the number of stacks in between each visualization',
        type=int, default=10)
    parser.add_argument(
        '--mask_smooth_radius',
        help='radius of average pooling kernel for smoothing smoothness '
        'penalty weights', type=int, default=20)
    parser.add_argument(
        '--mask_neighborhood_radius',
        help='radius (in pixels) of neighborhood effects of cracks and folds',
        type=int, default=20)
    parser.add_argument(
        '--eps',
        help='epsilon value to avoid divide-by-zero',
        type=float, default=1e-6)
    parser.add_argument(
        '--no_jitter',
        help='omit jitter when augmenting image stacks', action='store_true')
    parser.add_argument(
        '--hm',
        help='do high mip (mip 5) training run; runs at mip 2 if flag is '
        'omitted', action='store_true')
    parser.add_argument(
        '--mm',
        help='mix mips while training (2,5,7,8)', action='store_true')
    parser.add_argument(
        '--blank_var_threshold',
        help='variance threshold under which an image will be considered '
        'blank and omitted', type=float, default=0.001)
    parser.add_argument(
        '--lambda1', help='total smoothness penalty coefficient',
        type=float, default=0.1)
    parser.add_argument(
        '--lambda2', help='smoothness penalty reduction around cracks/folds',
        type=float, default=0.3)
    parser.add_argument(
        '--lambda3',
        help='smoothness penalty reduction on top of cracks/folds',
        type=float, default=0.00001)
    parser.add_argument(
        '--lambda4', help='MSE multiplier in regions around cracks and folds',
        type=float, default=10)
    parser.add_argument(
        '--lambda5',
        help='MSE multiplier in regions on top of cracks and folds',
        type=float, default=0)
    parser.add_argument(
        '--lambda6',
        help='coefficient for drift-prevention consensus loss in training '
        '(default=0)', type=float, default=0)
    parser.add_argument(
        '--skip',
        help='number of residuals (starting at the bottom of the pyramid) to '
        'omit from the aggregate field', type=int, default=0)
    parser.add_argument(
        '--size',
        help='height of pyramid/number of residual modules in the pyramid',
        type=int, default=5)
    parser.add_argument(
        '--dim',
        help='side dimension of training images', type=int, default=1152)
    parser.add_argument(
        '--trunc',
        help='truncation layer; should usually be size-1 (0 IF FINE TUNING)',
        type=int, default=4)
    parser.add_argument(
        '--lr',
        help='starting learning rate', type=float, default=0.0002)
    parser.add_argument(
        '--num_epochs',
        help='number of training epochs', type=int, default=1000)
    parser.add_argument(
        '--state_archive',
        help='saved model to initialize with', type=str, default=None)
    # parser.add_argument(
    #     '--batch_size',
    #     help='size of batch', type=int, default=1)
    parser.add_argument(
        '--k',
        help='kernel size', type=int, default=7)
    parser.add_argument(
        '--fall_time',
        help='epochs between layers', type=int, default=2)
    parser.add_argument(
        '--epoch',
        help='training epoch to start from', type=int, default=0)
    parser.add_argument(
        '--padding',
        help='amount of padding to add to training stacks',
        type=int, default=128)
    parser.add_argument(
        '--fine_tuning',
        help='when passed, begin with fine tuning (reduce learning rate, '
        'train all parameters)', action='store_true')
    parser.add_argument(
        '--skip_sample_aug',
        help='skips slice-wise augmentation when passed (no cutouts, etc)',
        action='store_true')
    parser.add_argument(
        '--penalty',
        help='type of smoothness penalty (lap, jacob, cjacob, tv)',
        type=str, default='jacob')
    parser.add_argument(
        '--lm_defect_net',
        help='mip2 defect net archive', type=str,
        default='basil_defect_unet18070201')
    parser.add_argument(
        '--hm_defect_net',
        help='mip5 defect net archive', type=str,
        default='basil_defect_unet_mip518070305')
    parser.add_argument(
        '--fine_tuning_lr_factor',
        help='factor by which to reduce learning rate during fine tuning',
        type=float, default=0.2)
    parser.add_argument(
        '--pe', action='store_true', 
        help="Add a preencoder to preprocess the inputs")
    parser.add_argument(
        '--pe_only', action='store_true',
        help="Only train the preencoder.")
    parser.add_argument(
        '--enc_only', action='store_true')
    parser.add_argument(
        '--vhm_src',
        help='very high mip source (mip 8)', type=str,
        default='~/matriarch_mixed.h5')
    parser.add_argument(
        '--hm_src',
        help='high mip source (mip 5)', type=str, default='~/mip5_mixed.h5')
    parser.add_argument(
        '--lm_src',
        help='low mip source (mip 2)', type=str, default='~/test_mip2_drosophila_cleaned.h5')
    parser.add_argument('--batch_size', type=int, default=1, 
        help='Number of samples to be evaluated before each gradient update')
    parser.add_argument('--num_workers', type=int, default=1,
        help='Number of workers for the DataLoader')
    parser.add_argument('--gpu_ids', type=str, default=['0'], nargs='+',
        help='Specific GPUs to use during training')
    return parser.parse_args()


if __name__ == '__main__':
    main()
