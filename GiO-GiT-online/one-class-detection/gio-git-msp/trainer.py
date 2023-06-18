import sys
sys.dont_write_bytecode = True
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn

import math
import scipy.io as io
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path

import matplotlib.pyplot as plt

from model import WideResNet, WideResNet_Sphere, WideResNet_Hyper, WideResNet_Mix
from dataloader import OneClassDatasetCIFAR10
from calculate_roc import detection_test
from torchsphere.nn import AngleLoss
from distance import Logits, SoftTarget
Distance = SoftTarget(T=1.0)


def main():
    # Parameters
    parser = argparse.ArgumentParser(description='Mix-geimetric-trans')
    parser.add_argument('--DATA_DIR', default='./cifar-10-batches-py', type=str)
    
    parser.add_argument('--NUM_CLASS', default=10, type=int)
    parser.add_argument('--NUM_OUTPUT', default=4, type=int)
    parser.add_argument('--EPOCHS', default=2001, type=int)
    parser.add_argument('--BATCH_SIZE', default=128, type=int)
    parser.add_argument('--VAL_EACH', default=10, type=int)
    parser.add_argument('--SAVE_EACH', default=500, type=int)

    parser.add_argument('--REAL_CLASS', default=0, type=int)   
    parser.add_argument('--USE_Baseline', default=False, action="store_true")
    parser.add_argument('--USE_Sphere', default=False, action="store_true")  
    parser.add_argument('--USE_Hyper',  default=False, action="store_true") 
    parser.add_argument('--USE_Mix',    default=False, action="store_true")

    parser.add_argument("--CURVATURE_Sphere", type=float, default=0.0)
    parser.add_argument("--CURVATURE_Hyper", type=float, default=0.0)

    args = parser.parse_args()

    print(args)

    if args.USE_Baseline:
        checkpoint_path = "./checkpoints_bsl/cifar10/{}/".format(args.REAL_CLASS)
    if args.USE_Sphere:
        checkpoint_path = "./checkpoints_sphere/cifar10/{}/".format(args.REAL_CLASS)
    if args.USE_Hyper:
        checkpoint_path = "./checkpoints_hyper/cifar10/{}/".format(args.REAL_CLASS)
    if args.USE_Mix:
        checkpoint_path = "./checkpoints_mix/cifar10/{}/".format(args.REAL_CLASS)
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    training_set = OneClassDatasetCIFAR10(args.DATA_DIR, real_class=args.REAL_CLASS, train=True)
    train_loader = DataLoader(training_set, batch_size=args.BATCH_SIZE)

    val_set      = OneClassDatasetCIFAR10(args.DATA_DIR, real_class=args.REAL_CLASS, train=False, vis=False)
    val_loader   = DataLoader(val_set, batch_size=args.BATCH_SIZE)

    criterion = nn.CrossEntropyLoss()
    criterion_s = AngleLoss()

    if args.USE_Baseline:
        model = WideResNet(num_of_classes=args.NUM_OUTPUT, depth=28, widen_factor=10)
    if args.USE_Sphere:
        model = WideResNet_Sphere(num_of_classes=args.NUM_OUTPUT, curvature=args.CURVATURE_Sphere, depth=28, widen_factor=10)
    if args.USE_Hyper:
        model = WideResNet_Hyper(num_of_classes=args.NUM_OUTPUT, curvature=args.CURVATURE_Hyper, depth=28, widen_factor=10)
    if args.USE_Mix:
        model = WideResNet_Mix(num_of_classes=args.NUM_OUTPUT, curvature_sphere=args.CURVATURE_Sphere, curvature_hyper=args.CURVATURE_Hyper, depth=28, widen_factor=10)

    model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    roc_auc_max  = 0.0
    curve_roc_auc = [] 
 
    for epoch_idx in range(args.EPOCHS):
        pbar = tqdm(train_loader)

        model.train()
        for image_batch, label_batch, _ in pbar:
 
            image_batch = image_batch.cuda().float()
            label_batch = label_batch.cuda()

            pbar.set_description("Epoch: %s" % str(epoch_idx))
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            if args.USE_Baseline:
                logits_batch, _, _ = model(image_batch)
                loss = criterion(logits_batch, label_batch)

            if args.USE_Sphere:
                logits_batch, logits_batch_sphere, _ = model(image_batch) 
                loss1 = criterion(logits_batch, label_batch)

                loss2 = criterion_s(logits_batch_sphere, label_batch)
                loss  = loss1 + loss2

            if args.USE_Hyper:
                logits_batch, logits_batch_hyper, _ = model(image_batch)
                loss1 = criterion(logits_batch, label_batch)
                loss2 = criterion(logits_batch_hyper, label_batch)
                loss  = loss1 + loss2

            if args.USE_Mix:
                logits_batch, logits_batch_sphere, logits_batch_hyper = model(image_batch)  
                loss1 = criterion(logits_batch, label_batch)
                loss2 = criterion_s(logits_batch_sphere, label_batch)
                loss3 = criterion(logits_batch_hyper, label_batch)
                loss  = loss1 + loss2 + loss3

            loss.backward()
            optimizer.step()

        model.eval()
        if epoch_idx % args.SAVE_EACH == 0:
            optimizer.zero_grad()
            torch.save(model.state_dict(), 
                       '{}Class_{}_epoch_{}.pth'.format(checkpoint_path, args.REAL_CLASS, epoch_idx))

        if epoch_idx % args.VAL_EACH == 0:
            correct = 0
            total = 0

            with torch.no_grad():
                pbar = tqdm(val_loader)
                for image_batch, label_batch, label_batch_real in pbar:
                    
                    image_batch = image_batch.cuda().float()
                    label_batch = label_batch.cuda()

                    logits_batch, _, _ = model(image_batch)

                    _, predicted = torch.max(logits_batch.data, 1)
                    total   += label_batch.size(0)
                    correct += (predicted == label_batch).sum().item()
                
                # calculate auroc
                roc_auc = detection_test(model, pbar, args)

                curve_roc_auc.append(roc_auc*100)
                io.savemat(os.path.join(checkpoint_path, 'roc_auc.mat'), {'roc_auc': np.array(curve_roc_auc)})

                plt.title('One-Class Detection on CIFAR-10')
                plt.plot(np.arange(len(curve_roc_auc)), curve_roc_auc)
                plt.ylim(0, 100)
                plt.xlabel('Epoch')
                plt.ylabel('AUROC (%)')
                plt.savefig(os.path.join(checkpoint_path, 'curve_roc_auc.jpg'))
                plt.close()

                if roc_auc > roc_auc_max: 
                    roc_auc_max = roc_auc        
                    optimizer.zero_grad()
                    torch.save(model.state_dict(), 
                               '{}Class_{}_best.pth'.format(checkpoint_path, args.REAL_CLASS))

                print('Epoch is: %d' % epoch_idx)
                print('Training loss is: %2f' % (float(loss)))   
                print('Rotation prediction accuracy on the 10000 test images, AUROC and AUROC(max) are: %d %%, %2f and %2f' % (100*correct/total, roc_auc, roc_auc_max))


main()
