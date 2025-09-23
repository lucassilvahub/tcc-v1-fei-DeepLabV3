from pathlib import Path
import numpy as np


def load_fold_files(data_path, fold_num):
    """
    Lê os arquivos de um fold específico (cross-validation).
    Retorna listas de caminhos de imagens e máscaras.
    """
    fold_path = Path(data_path) / "folds"

    images_file = fold_path / f"fold{fold_num}_images.txt"
    labels_file = fold_path / f"fold{fold_num}_labels.txt"

    with open(images_file, "r") as f:
        image_names = [line.strip() for line in f if line.strip()]
    with open(labels_file, "r") as f:
        label_names = [line.strip() for line in f if line.strip()]

    base_path = Path(data_path)
    image_paths = [base_path / "images" / name for name in image_names]
    label_paths = [base_path / "labels" / name for name in label_names]

    return image_paths, label_paths


def compute_iou(pred, target, num_classes):
    """
    Calcula Intersection-over-Union (IoU) por classe.
    """
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        if union == 0:
            ious.append(float("nan"))
        else:
            ious.append((intersection / union).item())
    return ious
