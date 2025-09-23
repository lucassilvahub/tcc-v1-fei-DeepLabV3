import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


class PetropolisDataset(Dataset):
    """
    Dataset para o problema de segmentação semântica de uso e cobertura do solo.
    Lê imagens RGB e converte máscaras coloridas em classes numéricas.
    """

    def __init__(self, image_paths, label_paths, config, mode="train"):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.config = config
        self.mode = mode

        # Definição das transformações de dados (augmentations)
        if mode == "train":
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            )
        else:  # validação/teste
            self.transform = A.Compose(
                [
                    A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            )

    def __len__(self):
        return len(self.image_paths)

    def rgb_to_class(self, rgb_mask):
        """
        Converte máscara colorida (RGB) para máscara de classes (0..N-1).
        """
        h, w = rgb_mask.shape[:2]
        class_mask = np.zeros((h, w), dtype=np.uint8)

        for class_id, color in self.config.CLASS_COLORS.items():
            mask = np.all(rgb_mask == color, axis=2)
            class_mask[mask] = class_id

        return class_mask

    def __getitem__(self, idx):
        # Carrega imagem
        image = cv2.imread(str(self.image_paths[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Carrega máscara
        mask = cv2.imread(str(self.label_paths[idx]))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = self.rgb_to_class(mask)

        # Aplica transformações
        transformed = self.transform(image=image, mask=mask)
        return transformed["image"], transformed["mask"].long()
