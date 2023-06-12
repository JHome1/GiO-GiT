# ------------------------------------------------------------------------------
# Modified from 'Anomaly-Detection-PatchSVDD-PyTorch' 
# Reference: https://github.com/nuclearboy95/Anomaly-Detection-PatchSVDD-PyTorch
# ------------------------------------------------------------------------------
import sys
sys.dont_write_bytecode = True
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import argparse
import torch
import statistics
from torch.utils.data import DataLoader
from functools import reduce

from codes import mvtecad
from codes.datasets import *
from codes.utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--obj',          default='hazelnut', type=str)
parser.add_argument('--lambda_value', default=1e-3, type=float)
parser.add_argument('--D',            default=64, type=int)
parser.add_argument('--D_emb',        default=128, type=int)

parser.add_argument('--epochs',    default=1201, type=int)
parser.add_argument('--log_epoch', default=20, type=int)
parser.add_argument('--lr',        default=1e-4, type=float)

parser.add_argument("--use_geo", default=None, type=str)
parser.add_argument("--curvature_s", default=0.0, type=float)
parser.add_argument("--curvature_h", default=0.0, type=float)
parser.add_argument("--use_minus_distance", default=False, action="store_true")
parser.add_argument("--use_max_distance",   default=False, action="store_true")
parser.add_argument("--use_det_64",         default=False, action="store_true")
args = parser.parse_args()

###########################################################
if args.use_geo == None: 
    from codes.networks import *
    from codes.inspection import eval_encoder_NN_multiK
elif args.use_geo == 'use_sphere': 
    from codes.networks_sphere import *
    from codes.inspection_sphere import eval_encoder_NN_multiK
elif args.use_geo == 'use_hyper': 
    from codes.networks_hyper import *
    from codes.inspection_hyper import eval_encoder_NN_multiK
elif args.use_geo == 'use_mix': 
    from codes.networks_mix import *
    from codes.inspection_mix import eval_encoder_NN_multiK
###########################################################   

print(args)


def train():
    obj   = args.obj
    D     = args.D
    D_emb = args.D_emb
    lr    = args.lr

    curve_seg_64 = []
    curve_seg_32 = []
    curve_seg_sum  = []
    curve_seg_mult = []      

    loss = 0.0 

    with task('Networks'):
        enc = EncoderHier(64, D, D_emb).cuda()

        ########################
        if args.use_geo == None:
            cls_64 = PositionClassifier(64, D, D_emb).cuda()
            cls_32 = PositionClassifier(32, D, D_emb).cuda()
            
        elif args.use_geo == 'use_sphere':
            cls_64 = PositionClassifier(64, D, D_emb, curvature=args.curvature_s).cuda()
            cls_32 = PositionClassifier(32, D, D_emb, curvature=args.curvature_s).cuda()

        elif args.use_geo == 'use_hyper':
            cls_64 = PositionClassifier(64, D, D_emb, curvature=args.curvature_h).cuda()
            cls_32 = PositionClassifier(32, D, D_emb, curvature=args.curvature_h).cuda()

        elif args.use_geo == 'use_mix':
            cls_64 = PositionClassifier(64, D, D_emb, curvature_s=args.curvature_s, curvature_h=args.curvature_h).cuda()
            cls_32 = PositionClassifier(32, D, D_emb, curvature_s=args.curvature_s, curvature_h=args.curvature_h).cuda()
        ################################################################################################################

        modules = [enc, cls_64, cls_32]
        params = [list(module.parameters()) for module in modules]
        params = reduce(lambda x, y: x + y, params)

        opt = torch.optim.Adam(params=params, lr=lr)

    with task('Datasets'):
        train_x = mvtecad.get_x_standardized(obj, mode='train')
        train_x = NHWC2NCHW(train_x)

        rep = 100
        datasets = dict()
        datasets[f'pos_64'] = PositionDataset(train_x, K=64, repeat=rep)
        datasets[f'pos_32'] = PositionDataset(train_x, K=32, repeat=rep)
        
        datasets[f'svdd_64'] = SVDD_Dataset(train_x, K=64, repeat=rep)
        datasets[f'svdd_32'] = SVDD_Dataset(train_x, K=32, repeat=rep)

        dataset = DictionaryConcatDataset(datasets)
        loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)

    print('Start training')
    for i_epoch in range(args.epochs):
        if i_epoch != 0:
            for module in modules:
                module.train()

            for d in loader:
                d = to_device(d, 'cuda', non_blocking=True)
                opt.zero_grad()

                loss_pos_64 = PositionClassifier.infer(cls_64, enc, d['pos_64'])
                loss_pos_32 = PositionClassifier.infer(cls_32, enc.enc, d['pos_32'])
                loss_svdd_64 = SVDD_Dataset.infer(enc, d['svdd_64'])
                loss_svdd_32 = SVDD_Dataset.infer(enc.enc, d['svdd_32'])

                loss = loss_pos_64 + loss_pos_32 + args.lambda_value * (loss_svdd_64 + loss_svdd_32)

                loss.backward()
                opt.step()
        
        print('Epoch: %d and Training Loss: %2f' %(i_epoch, float(loss)))
        if i_epoch%args.log_epoch==0 and i_epoch!=0:   
            aurocs = eval_encoder_NN_multiK(enc, cls_32, cls_64, obj, args)
        
            seg_64   = aurocs['seg_64'] * 100
            seg_32   = aurocs['seg_32'] * 100
            seg_sum  = aurocs['seg_sum'] * 100
            seg_mult = aurocs['seg_mult'] * 100

            curve_seg_64.append(seg_64)
            curve_seg_32.append(seg_32)
            curve_seg_sum.append(seg_sum)
            curve_seg_mult.append(seg_mult)

            log_result(i_epoch, loss, obj,
                       seg_64,
                       seg_32,
                       seg_sum,
                       seg_mult)
            
            enc.save_curve(obj,
                           curve_seg_64,
                           curve_seg_32,
                           curve_seg_sum,
                           curve_seg_mult)


def log_result(i_epoch, loss, obj,
               seg_64,
               seg_32,
               seg_sum,
               seg_mult,
):
    print(f'|K64| Seg: {seg_64:4.1f} |K32| Seg: {seg_32:4.1f} |sum| Seg: {seg_sum:4.1f} |mult| Seg: {seg_mult:4.1f} ({obj})')


if __name__ == '__main__':
    train()
