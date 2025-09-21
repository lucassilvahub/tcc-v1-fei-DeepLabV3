import time
import torch
import os
import cv2
import numpy as np
from skimage import io, img_as_float64
from project_utils import convert_from_color, convert_to_color
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

EXTS = ['tif', 'tiff', 'jpg', 'png']

# Dataset class with sliding-window cropping, supports tuple/int for window_size and stride
class DatasetIcmbio(torch.utils.data.Dataset):

    def __init__(
        self,
        data_files,
        label_files,
        edge_files=None,
        window_size=256,
        stride=64,
        n_channels=3,
        cache=True,
        augmentation=True
    ):
        super().__init__()
        self.data_files   = data_files
        self.label_files  = label_files
        self.edge_files   = edge_files
        # support both int and tuple window_size
        if isinstance(window_size, tuple):
            self.window_h, self.window_w = window_size
        else:
            self.window_h = self.window_w = window_size
        # support both int and tuple stride
        if isinstance(stride, tuple):
            self.stride_h, self.stride_w = stride
        else:
            self.stride_h = self.stride_w = stride
        self.n_channels   = n_channels
        self.cache        = cache
        self.augmentation = augmentation

        # sanity check files exist
        all_files = self.data_files + self.label_files
        if self.edge_files is not None:
            all_files += self.edge_files
        for f in all_files:
            if not os.path.isfile(f):
                raise KeyError(f"'{f}' nao encontrado!")

        # caches
        self.data_cache_  = {}
        self.label_cache_ = {}
        self.edge_cache_  = {}

        # determine patch grid for the first image (assumes all images same size)
        sample = io.imread(self.data_files[0])
        H, W = sample.shape[:2]
        # calculate number of steps using floor((I-C)/s)+1
        self.steps_h = (H - self.window_h) // self.stride_h + 1
        self.steps_w = (W - self.window_w) // self.stride_w + 1
        self.patches_per_img = self.steps_h * self.steps_w

    def __len__(self):
        return len(self.data_files) * self.patches_per_img

    def get_dataset(self):
        return self.data_files, self.label_files, self.edge_files

    @classmethod
    def data_augmentation(cls, *arrays, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        if flip and np.random.rand() < 0.5:
            will_flip = True
        if mirror and np.random.rand() < 0.5:
            will_mirror = True

        results = []
        for array in arrays:
            if will_flip:
                array = array[..., ::-1]
            if will_mirror:
                array = array[..., :, ::-1]
            results.append(np.copy(array))
        return tuple(results)

    def __getitem__(self, idx):
        img_idx    = idx // self.patches_per_img
        patch_idx  = idx % self.patches_per_img
        row        = patch_idx // self.steps_w
        col        = patch_idx % self.steps_w
        x1 = row * self.stride_h
        y1 = col * self.stride_w
        x2 = x1 + self.window_h
        y2 = y1 + self.window_w

        # load or cache image data
        if img_idx in self.data_cache_:
            data = self.data_cache_[img_idx]
        else:
            img = io.imread(self.data_files[img_idx])
            data = (img.transpose(2,0,1) / 255.0).astype('float32')
            if self.cache:
                self.data_cache_[img_idx] = data

        # load or cache label data
        if img_idx in self.label_cache_:
            label = self.label_cache_[img_idx]
        else:
            lbl_img = io.imread(self.label_files[img_idx])[...,:3]
            label = convert_from_color(lbl_img).astype('int64')
            if self.cache:
                self.label_cache_[img_idx] = label

        # load or cache edges if provided
        if self.edge_files is not None:
            if img_idx in self.edge_cache_:
                edges = self.edge_cache_[img_idx]
            else:
                edge_img = io.imread(self.edge_files[img_idx])
                if edge_img.ndim == 3:
                    edges = (edge_img[..., 0] / 255.0).astype('float32')
                elif edge_img.ndim == 2:
                    edges = (edge_img / 255.0).astype('float32')
                else:
                    raise ValueError(f"Unexpected edge_img.ndim = {edge_img.ndim}")
                if self.cache:
                    self.edge_cache_[img_idx] = edges
        else:
            edges = None

        # extract patch
        data_p = data[:, x1:x2, y1:y2]
        label_p = label[x1:x2, y1:y2]
        if edges is not None:
            edge_p = edges[x1:x2, y1:y2]

        # apply augmentation
        if self.augmentation:
            if edges is not None:
                data_p, label_p, edge_p = self.data_augmentation(data_p, label_p, edge_p)
            else:
                data_p, label_p = self.data_augmentation(data_p, label_p)

        # convert to tensor
        data_t  = torch.from_numpy(data_p)
        label_t = torch.from_numpy(label_p)
        if edges is not None:
            edge_t = torch.from_numpy(edge_p)
            return data_t, label_t, edge_t
        return data_t, label_t
