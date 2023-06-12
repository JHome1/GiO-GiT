# ------------------------------------------------------------------------------
# Modified from 'Anomaly-Detection-PatchSVDD-PyTorch' 
# Reference: https://github.com/nuclearboy95/Anomaly-Detection-PatchSVDD-PyTorch
# ------------------------------------------------------------------------------
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from codes     import mvtecad
from .utils    import PatchDataset_NCHW, NHWC2NCHW, distribute_scores
from .distance import Logits, SoftTarget

__all__ = ['eval_encoder_NN_multiK', 'eval_embeddings_NN_multiK']
KL_distance = SoftTarget(T=1.0)


def infer(x, enc, cls, K, S, args):
    x = NHWC2NCHW(x)
    dataset = PatchDataset_NCHW(x, K=K, S=S)
    loader  = DataLoader(dataset, batch_size=2048, shuffle=False, pin_memory=True)
   
    d_se = np.empty((dataset.N, dataset.row_num, dataset.col_num), dtype=np.float32)

    enc = enc.eval()
    cls = cls.eval()

    with torch.no_grad():
        for xs, x_jit, ns, iis, js in loader:

            xs     = xs.cuda()
            xs_jit = x_jit.float().cuda()     
           
            embedding     = enc(xs)
            embedding_jit = enc(xs_jit)

            logit, logit_s, _, _, _, _ = cls(embedding, embedding_jit)
            logit_s = logit_s[0]        

            d_se_tmp = torch.zeros(logit.shape[0])
            for i in range(logit.shape[0]):
                d_se_ele = KL_distance(logit_s[i].unsqueeze(0), logit[i].unsqueeze(0))
                d_se_ele = torch.tanh(d_se_ele)
                d_se_tmp[i] = 1-d_se_ele # embedding: d_se_ele; logit: 1-d_se_ele
                if not args.use_minus_distance:
                    d_se_tmp[i] = d_se_ele

            d_se_tmp = d_se_tmp.detach().cpu().numpy()

            for d_se_ele, n, i, j in zip(d_se_tmp, ns, iis, js):
                d_se[n, i, j] = d_se_ele
    return d_se


def assess_anomaly_maps(obj, anomaly_maps, args):
    auroc_seg      = mvtecad.segmentation_auroc(obj, anomaly_maps)

    anomaly_scores = anomaly_maps.mean(axis=-1).mean(axis=-1)
    if args.use_max_distance: 
        anomaly_scores = anomaly_maps.max(axis=-1).max(axis=-1)

    auroc_det = mvtecad.detection_auroc(obj, anomaly_scores)

    return auroc_det, auroc_seg


def eval_encoder_NN_multiK(enc, cls_32, cls_64, obj, args):
    x_te = mvtecad.get_x_standardized(obj, mode='test')

    if args.use_det_64:
        d_64_te = infer(x_te, enc, cls_64, K=64, S=16, args=args)
    else: d_64_te = 0.0

    d_32_te = infer(x_te, enc.enc, cls_32, K=32, S=4, args=args)

    d_64 = d_64_te
    d_32 = d_32_te

    return eval_embeddings_NN_multiK(obj, d_64, d_32, args)


def eval_embeddings_NN_multiK(obj, d_64, d_32, args, NN=1):
    if args.use_det_64:
        d_te = d_64
        maps_64 = distribute_scores(d_te, (256, 256), K=64, S=16)
        det_64, seg_64 = assess_anomaly_maps(obj, maps_64, args)
    else: 
        det_64  = 0.5
        seg_64  = 0.5
        maps_64 = 0.0

    d_te = d_32
    maps_32 = distribute_scores(d_te, (256, 256), K=32, S=4)
    det_32, seg_32 = assess_anomaly_maps(obj, maps_32, args)

    if args.use_det_64:
        maps_sum = maps_64 + maps_32
        det_sum, seg_sum = assess_anomaly_maps(obj, maps_sum, args)

        maps_mult = maps_64 * maps_32
        det_mult, seg_mult = assess_anomaly_maps(obj, maps_mult, args)
    else:
        det_sum   = 0.5
        seg_sum   = 0.5
        maps_sum  = 0.0
        det_mult  = 0.5
        seg_mult  = 0.5
        maps_mult = 0.0

    return {
            'det_64': det_64,
            'seg_64': seg_64,

            'det_32': det_32,
            'seg_32': seg_32,

            'det_sum': det_sum,
            'seg_sum': seg_sum,

            'det_mult': det_mult,
            'seg_mult': seg_mult,

            'maps_64': maps_64,
            'maps_32': maps_32,
            'maps_sum': maps_sum,
            'maps_mult': maps_mult,
            }


def measure_emb_NN(emb_te, emb_tr, method='kdt', NN=1):
    from .nearest_neighbor import search_NN
    D = emb_tr.shape[-1]
    train_emb_all = emb_tr.reshape(-1, D)

    l2_maps, _   = search_NN(emb_te, train_emb_all, method=method, NN=NN)
    anomaly_maps = np.mean(l2_maps, axis=-1)

    return anomaly_maps
