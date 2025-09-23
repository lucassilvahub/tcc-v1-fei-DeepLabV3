import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

class PetropolisPatchDataset(Dataset):
    def __init__(self, image_paths, label_paths, config, patch_size=256, stride=128, mode="train"):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.config = config
        self.patch_size = patch_size
        self.stride = stride
        self.mode = mode
        self.patches = []
        self._generate_patches()

        if mode == "train":
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.GaussianBlur(blur_limit=(3,5), p=0.3),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5),
                A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
                ToTensorV2(),
            ])

    def _generate_patches(self):
        for img_path, lbl_path in zip(self.image_paths, self.label_paths):
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(str(lbl_path))
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            mask = self.rgb_to_class(mask)

            H, W = image.shape[:2]
            for i in range(0, H - self.patch_size + 1, self.stride):
                for j in range(0, W - self.patch_size + 1, self.stride):
                    img_patch = image[i:i+self.patch_size, j:j+self.patch_size]
                    mask_patch = mask[i:i+self.patch_size, j:j+self.patch_size]
                    self.patches.append((img_patch, mask_patch))

    def rgb_to_class(self, rgb_mask):
        h, w = rgb_mask.shape[:2]
        class_mask = np.zeros((h, w), dtype=np.uint8)
        for class_id, color in self.config.CLASS_COLORS.items():
            mask = np.all(rgb_mask == color, axis=2)
            class_mask[mask] = class_id
        return class_mask

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        image, mask = self.patches[idx]
        transformed = self.transform(image=image, mask=mask)
        return transformed["image"], transformed["mask"].long()
