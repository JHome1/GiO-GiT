import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import ReLU
from copy import deepcopy

from sklearn.metrics import roc_curve, auc
from scipy.ndimage.filters import gaussian_filter
from distance import Logits, SoftTarget

Distance = SoftTarget(T=1.0)


def detection_test(model, pbar, args):
    list_Z      = []
    list_score  = []
    model.eval()

    for image_batch, label_batch, label_real_batch in pbar:
        X, Y, Z = image_batch, label_batch, label_real_batch
        X = Variable(X).cuda().float()

        if args.USE_Baseline:
            logits_batch, _, _ = model.forward(X)

        if args.USE_Sphere:
            logits_batch, logits_batch_sphere, _ = model.forward(X)
            logits_batch_sphere = logits_batch_sphere[0]
  
        if args.USE_Hyper:
            logits_batch, logits_batch_hyper, _  = model.forward(X)

        if args.USE_Mix:
            logits_batch, logits_batch_sphere, logits_batch_hyper = model.forward(X)
            logits_batch_sphere = logits_batch_sphere[0]

        _, predicted      = torch.max(logits_batch.data, 1)
        predicted_conf, _ = torch.max(F.softmax(logits_batch, dim=1), 1)

        for i in range(0, Y.size(0)):
            if args.USE_Baseline:
                score = float(1-predicted_conf[i])                

            if args.USE_Sphere:
                d_ls = Distance(logits_batch_sphere[i].unsqueeze(0), logits_batch[i].unsqueeze(0))
                score = float(torch.tanh(d_ls))
  
            if args.USE_Hyper:
                d_lh = Distance(logits_batch_hyper[i].unsqueeze(0), logits_batch[i].unsqueeze(0)) 
                score = float(torch.tanh(d_lh))

            if args.USE_Mix:
                d_ls = Distance(logits_batch_sphere[i].unsqueeze(0), logits_batch[i].unsqueeze(0)) 
                d_lh = Distance(logits_batch_hyper[i].unsqueeze(0), logits_batch[i].unsqueeze(0))
                d_lm  = math.sqrt(d_ls**2 + d_lh**2)
                d_lm  = torch.tensor(d_lm)
                score = float(torch.tanh(d_lm))

            if predicted[i] != Y[i]:
                score = 1.0

            list_Z.append(Z[i])
            list_score.append(score) # embedding: 1-score; logit: score

    labels = np.array(list_Z)
    indx1 = labels == args.REAL_CLASS
    indx2 = labels != args.REAL_CLASS
    labels[indx1] = 1
    labels[indx2] = 0
    scores = np.array(list_score)
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=0)
    roc_auc = auc(fpr, tpr)
    roc_auc = round(roc_auc, 4)
    return roc_auc
