# ------------------------------------------------------------------------------
# Modified from 'Anomaly-Detection-PatchSVDD-PyTorch' 
# Reference: https://github.com/nuclearboy95/Anomaly-Detection-PatchSVDD-PyTorch
# ------------------------------------------------------------------------------
import os
import _pickle as p

import numpy as np
import torch
from torch.utils.data import Dataset

from contextlib import contextmanager
from PIL import Image

__all__ = ['crop_image_CHW', 'PatchDataset_NCHW', 'NHWC2NCHW_normalize', 'NHWC2NCHW',
           'save_binary', 'load_binary', 'makedirpath', 'task', 'DictionaryConcatDataset',
           'to_device', 'distribute_scores', 'resize']

pos_to_diff = {0: (-1, -1),
               1: (-1, 0),
               2: (-1, 1),
               3: (0, -1),
               4: (0, 1),
               5: (1, -1),
               6: (1, 0),
               7: (1, 1)
               }


def to_device(obj, device, non_blocking=False):
    """
    :param obj:
    :param device:
    :param bool non_blocking:
    :return:
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=non_blocking)

    if isinstance(obj, dict):
        return {k: to_device(v, device, non_blocking=non_blocking)
                for k, v in obj.items()}

    if isinstance(obj, list):
        return [to_device(v, device, non_blocking=non_blocking)
                for v in obj]

    if isinstance(obj, tuple):
        return tuple([to_device(v, device, non_blocking=non_blocking)
                     for v in obj])


@contextmanager
def task(_):
    yield


class DictionaryConcatDataset(Dataset):
    def __init__(self, d_of_datasets):
        self.d_of_datasets = d_of_datasets
        lengths = [len(d) for d in d_of_datasets.values()]
        self._length = min(lengths)
        self.keys = self.d_of_datasets.keys()
        assert min(lengths) == max(lengths), 'Length of the datasets should be the same'

    def __getitem__(self, idx):
        return {
            key: self.d_of_datasets[key][idx]
            for key in self.keys
        }

    def __len__(self):
        return self._length


def crop_CHW(image, i, j, K, S=1):
    if S == 1:
        h, w = i, j
    else:
        h = S * i
        w = S * j
    return image[:, h: h + K, w: w + K]


def crop_CHW_jit(image, i, j, H, W, K, S=1):
    if S == 1:
        h, w = i, j
    else:
        h = S * i
        w = S * j

    # get jittered patch
    J = K // 4

    K3_4 = 3 * K // 4
    pos = np.random.randint(8)
    h_dir, w_dir = pos_to_diff[pos]
    h_del, w_del = np.random.randint(J, size=2)

    h_diff = h_dir * (h_del + K3_4)
    w_diff = w_dir * (w_del + K3_4)

    h = h + h_diff
    w = w + w_diff

    h = np.clip(h, 0, H - K)
    w = np.clip(w, 0, W - K)

    patch_jit = image[:, h: h + K, w: w + K] + np.random.normal(scale=0.02, size=(3, 1, 1))
    patch_jit = patch_jit + np.random.normal(scale=0.02, size=(3, K, K))

    return patch_jit


def cnn_output_size(H, K, S=1, P=0) -> int:
    """
    :param int H: input_size
    :param int K: filter_size
    :param int S: stride
    :param int P: padding
    :return:
    """
    return 1 + (H - K + 2 * P) // S


def crop_image_CHW(image, coord, K):
    h, w = coord
    return image[:, h: h + K, w: w + K]


class PatchDataset_NCHW(Dataset):
    def __init__(self, memmap, tfs=None, K=32, S=1):
        super().__init__()
        self.arr = memmap
        self.tfs = tfs
        self.S = S
        self.K = K
        self.N = self.arr.shape[0]

    def __len__(self):
        return self.N * self.row_num * self.col_num

    @property
    def row_num(self):
        N, C, H, W = self.arr.shape
        K = self.K
        S = self.S
        I = cnn_output_size(H, K=K, S=S)
        return I

    @property
    def col_num(self):
        N, C, H, W = self.arr.shape
        K = self.K
        S = self.S
        J = cnn_output_size(W, K=K, S=S)
        return J

    def __getitem__(self, idx):
        N = self.N
        n, i, j = np.unravel_index(idx, (N, self.row_num, self.col_num))
        K = self.K
        S = self.S
        image = self.arr[n]
        patch     = crop_CHW(image, i, j, K, S)
        patch_jit = crop_CHW_jit(image, i, j, 256, 256, K, S)

        if self.tfs:
            patch     = self.tfs(patch)
            patch_jit = self.tfs(patch_jit)

        return patch, patch_jit, n, i, j


def NHWC2NCHW_normalize(x):
    x = (x / 255.).astype(np.float32)
    return np.transpose(x, [0, 3, 1, 2])


def NHWC2NCHW(x):
    return np.transpose(x, [0, 3, 1, 2])


def load_binary(fpath, encoding='ASCII'):
    with open(fpath, 'rb') as f:
        return p.load(f, encoding=encoding)


def save_binary(d, fpath):
    with open(fpath, 'wb') as f:
        p.dump(d, f)


def makedirpath(fpath: str):
    # dpath = os.path.dirname(fpath)
    # if dpath:
    #     os.makedirs(dpath, exist_ok=True)
    os.makedirs(fpath, exist_ok=True)


def distribute_scores(score_masks, output_shape, K: int, S: int) -> np.ndarray:
    N = score_masks.shape[0]
    results = [distribute_score(score_masks[n], output_shape, K, S) for n in range(N)]
    return np.asarray(results)


def distribute_score(score_mask, output_shape, K: int, S: int) -> np.ndarray:
    H, W = output_shape
    mask = np.zeros([H, W], dtype=np.float32)
    cnt  = np.zeros([H, W], dtype=np.int32)

    I, J = score_mask.shape[:2]
    for i in range(I):
        for j in range(J):
            h, w = i * S, j * S

            mask[h: h + K, w: w + K] += score_mask[i, j]
            cnt[h: h + K, w: w + K] += 1

    cnt[cnt == 0] = 1

    return mask / cnt


def resize(image, shape=(256, 256)):
    return np.array(Image.fromarray(image).resize(shape[::-1]))
