import torch
import numpy as np
from pathlib import Path
import cv2

# Diretórios globais
ROOT = Path(__file__).resolve().parent.parent   # sobe da pasta src/ para raiz
OUTPUTS_DIR = ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


def compute_iou(pred, target, num_classes):
    """
    Calcula IoU (Intersection over Union) para cada classe e o mIoU.
    Args:
        pred (torch.Tensor): predições [H, W] ou [N, H, W]
        target (torch.Tensor): rótulos verdadeiros [H, W] ou [N, H, W]
        num_classes (int): número de classes
    Returns:
        (list, float): lista de IoUs por classe, mIoU
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

    # mIoU ignora NaN (classes ausentes no batch)
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    miou = np.mean(valid_ious) if valid_ious else 0.0
    return ious, miou


def print_ious(ious, class_names):
    """
    Exibe IoU por classe no terminal.
    """
    print("\n📊 IoU por classe:")
    for name, iou in zip(class_names, ious):
        if np.isnan(iou):
            print(f"  {name:15}:   ---")
        else:
            print(f"  {name:15}: {iou*100:6.2f}%")


def load_fold_files(data_path, fold_num):
    """
    Lê os arquivos de um fold específico (cross-validation).
    Retorna listas de caminhos de imagens e máscaras.
    """
    fold_path = Path(data_path) / "folds"
    images_file = fold_path / f"fold{fold_num}_images.txt"
    labels_file = fold_path / f"fold{fold_num}_labels.txt"

    if not images_file.exists() or not labels_file.exists():
        raise FileNotFoundError(f"❌ Arquivos do fold {fold_num} não encontrados em {fold_path}")

    with open(images_file, "r") as f:
        image_names = [line.strip() for line in f if line.strip()]
    with open(labels_file, "r") as f:
        label_names = [line.strip() for line in f if line.strip()]

    base_path = Path(data_path)
    image_paths = [base_path / "images" / name for name in image_names]
    label_paths = [base_path / "labels" / name for name in label_names]

    return image_paths, label_paths


def save_colored_mask(mask, class_colors, output_path):
    """
    Salva uma máscara de predição em formato colorido (PNG).
    Converte RGB -> BGR antes de salvar no OpenCV.
    """
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in class_colors.items():
        # converter RGB -> BGR (OpenCV espera BGR)
        bgr = (color[2], color[1], color[0])
        color_mask[mask == class_id] = bgr

    output_path = Path(output_path).resolve()  # caminho absoluto
    ok = cv2.imwrite(str(output_path), color_mask)

    if not ok:
        raise RuntimeError(f"❌ Erro ao salvar {output_path}")
    else:
        print(f"✅ Máscara salva em {output_path}")
