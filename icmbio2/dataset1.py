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
        stride=128,
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
    def data_augmentation(cls, *arrays, p_flip: float = 0.5, p_mirror: float = 0.5):
        """
        Simple augmentation: random horizontal and vertical flips on CHW arrays.
        """
        do_flip = np.random.rand() < p_flip
        do_mirror = np.random.rand() < p_mirror
        results = []
        for arr in arrays:
            out = arr
            if do_flip:
                out = out[..., ::-1]
            if do_mirror:
                out = out[..., ::-1, :]
            results.append(np.ascontiguousarray(out))
        return tuple(results)

    @classmethod
    def data_augmentation_strong(
        cls,
        data_p: np.ndarray,
        label_p: np.ndarray,
        edge_p: np.ndarray = None,
        p_flip: float = 0.5,
        p_mirror: float = 0.5,
        p_rotate: float = 0.5,
        crop_size: tuple = (256, 256),
        p_crop: float = 0.5,
        p_elastic: float = 0.5,
        elastic_alpha: float = 1.0,
        elastic_sigma: float = 50.0,
        p_color: float = 0.5,
        brightness_limit: float = 0.2,
        contrast_limit: float = 0.2,
        p_noise: float = 0.5,
        noise_var: tuple = (10.0, 50.0),
        noise_mean: float = 0.0,
        p_blur: float = 0.5,
        blur_limit: int = 7,
        p_dropout: float = 0.5,
        dropout_holes: int = 8,
        dropout_height: int = 32,
        dropout_width: int = 32,
        dropout_fill: float = 0.0
    ) -> tuple:
        """
        Strong augmentations applied on CHW float arrays in [0,1], avoiding transposes and dtype changes.
        data_p: float32 CHW, label_p: int64 HW, edge_p: float32 HW
        Returns same shapes and dtypes.
        """
        # ensure contiguous
        img = np.copy(data_p)
        lbl = np.copy(label_p)
        edg = np.copy(edge_p) if edge_p is not None else None
        C, H, W = img.shape

        # Flip horizontal (width axis)
        if np.random.rand() < p_flip:
            img = img[..., :, ::-1]
            lbl = lbl[:, ::-1]
            if edg is not None:
                edg = edg[:, ::-1]

        # Flip vertical (height axis)
        if np.random.rand() < p_mirror:
            img = img[..., ::-1, :]
            lbl = lbl[::-1, :]
            if edg is not None:
                edg = edg[::-1, :]

        # Rotate 90-degree multiples only to avoid interpolation dtype issues
        if np.random.rand() < p_rotate:
            k = np.random.choice([1, 2, 3])
            img = np.rot90(img, k, axes=(1,2))
            lbl = np.rot90(lbl, k)
            if edg is not None:
                edg = np.rot90(edg, k)

        # Random Crop
        if np.random.rand() < p_crop:
            y0 = np.random.randint(0, H - crop_size[0] + 1)
            x0 = np.random.randint(0, W - crop_size[1] + 1)
            img = img[:, y0:y0+crop_size[0], x0:x0+crop_size[1]]
            lbl = lbl[y0:y0+crop_size[0], x0:x0+crop_size[1]]
            if edg is not None:
                edg = edg[y0:y0+crop_size[0], x0:x0+crop_size[1]]

        # Color jitter (brightness & contrast)
        if np.random.rand() < p_color:
            factor = 1.0 + np.random.uniform(-contrast_limit, contrast_limit)
            img = np.clip(img * factor, 0, 1)
            shift = np.random.uniform(-brightness_limit, brightness_limit)
            img = np.clip(img + shift, 0, 1)

        # Gaussian noise
        if np.random.rand() < p_noise:
            var = np.random.uniform(noise_var[0]/255.0, noise_var[1]/255.0)
            noise = np.random.normal(noise_mean, np.sqrt(var), img.shape).astype(img.dtype)
            img = np.clip(img + noise, 0, 1)

        # Blur (simple box blur)
        if np.random.rand() < p_blur:
            k = blur_limit if blur_limit % 2 == 1 else blur_limit+1
            img = cv2.blur(img.transpose(1,2,0), (k, k)).transpose(2,0,1)

        # Coarse Dropout
        if np.random.rand() < p_dropout:
            for _ in range(dropout_holes):
                y1 = np.random.randint(0, crop_size[0] - dropout_height)
                x1 = np.random.randint(0, crop_size[1] - dropout_width)
                img[:, y1:y1+dropout_height, x1:x1+dropout_width] = dropout_fill

        # ensure contiguous
        img = np.ascontiguousarray(img)
        lbl = np.ascontiguousarray(lbl)
        if edg is not None:
            edg = np.ascontiguousarray(edg)

        if edg is not None:
            return img, lbl, edg
        return img, lbl

    def __getitem__(self, idx):
        # calculate patch indices
        img_idx = idx // self.patches_per_img
        patch_idx = idx % self.patches_per_img
        row = patch_idx // self.steps_w
        col = patch_idx % self.steps_w
        y1 = row * self.stride_h
        x1 = col * self.stride_w
        y2 = y1 + self.window_h
        x2 = x1 + self.window_w

        # load image as float32 CHW
        if img_idx in self.data_cache_:
            data = self.data_cache_[img_idx]
        else:
            img = io.imread(self.data_files[img_idx])[:, :, :3] / 255.0
            data = img.transpose(2,0,1).astype('float32')
            if self.cache:
                self.data_cache_[img_idx] = data

        # load label as int64 HW
        if img_idx in self.label_cache_:
            label = self.label_cache_[img_idx]
        else:
            lbl_img = io.imread(self.label_files[img_idx])[:,:,:3]
            label = convert_from_color(lbl_img).astype('int64')
            if self.cache:
                self.label_cache_[img_idx] = label

        # load edges as float32 HW
        if self.edge_files is not None:
            if img_idx in self.edge_cache_:
                edge = self.edge_cache_[img_idx]
            else:
                edge_img = io.imread(self.edge_files[img_idx])
                if edge_img.ndim == 3:
                    edge = (edge_img[...,0] / 255.0).astype('float32')
                else:
                    edge = (edge_img / 255.0).astype('float32')
                if self.cache:
                    self.edge_cache_[img_idx] = edge
        else:
            edge = None

        # extract patch (CHW and HW)
        data_p = data[:, y1:y2, x1:x2]
        label_p = label[y1:y2, x1:x2]
        edge_p = edge[y1:y2, x1:x2] if edge is not None else None

        # apply augmentation
        if self.augmentation:
            if edge_p is not None:
                data_p, label_p, edge_p = self.data_augmentation_strong(data_p, label_p, edge_p)
            else:
                data_p, label_p = self.data_augmentation_strong(data_p, label_p)

        # convert to torch tensors
        data_t = torch.from_numpy(data_p)
        label_t = torch.from_numpy(label_p)
        if edge_p is not None:
            edge_t = torch.from_numpy(edge_p)
            return data_t, label_t, edge_t
        return data_t, label_t

