# ========================================
# PROJETO SEGMENTAÇÃO SEMÂNTICA - K-FOLD COM ARQUIVOS FOLD
# ========================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from torchmetrics import Accuracy, F1Score, JaccardIndex
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# ========================================
# CONFIGURAÇÃO
# ========================================

CONFIG = {
    "data_path": "./",  # Pasta que contém folds/, images/, labels/
    "batch_size": 8,
    "num_classes": 8,
    "image_size": (256, 256),
    "epochs": 10,  # Mais épocas para convergir melhor
    "lr": 1e-3,  # Learning rate
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# Cores das classes (RGB)
CLASS_COLORS = {
    0: (255, 0, 0),  # Urbano - Vermelho
    1: (0, 255, 0),  # Vegetação Densa - Verde
    2: (0, 0, 0),  # Sombra - Preto
    3: (255, 255, 0),  # Vegetação Esparsa - Amarelo
    4: (255, 165, 0),  # Agricultura - Laranja
    5: (128, 128, 128),  # Rocha - Cinza
    6: (139, 69, 19),  # Solo Exposto - Marrom
    7: (0, 0, 255),  # Água - Azul
}

CLASS_NAMES = [
    "Urbano",
    "Vegetação Densa",
    "Sombra",
    "Vegetação Esparsa",
    "Agricultura",
    "Rocha",
    "Solo Exposto",
    "Água",
]

# ========================================
# FUNÇÕES UTILITÁRIAS
# ========================================


def rgb_to_class(rgb_image):
    """Converte label RGB (anotação manual) para mapa de classes numéricas"""
    h, w = rgb_image.shape[:2]
    class_map = np.zeros((h, w), dtype=np.uint8)

    for class_id, color in CLASS_COLORS.items():
        mask = np.all(rgb_image == color, axis=2)
        class_map[mask] = class_id

    return class_map


def class_to_rgb(class_map):
    """Converte mapa de classes para RGB (para visualização das predições)"""
    h, w = class_map.shape
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in CLASS_COLORS.items():
        mask = class_map == class_id
        rgb_image[mask] = color

    return rgb_image


def load_fold_files(data_path, fold_num):
    """Carrega listas de arquivos de um fold específico"""
    fold_path = Path(data_path) / "folds"

    # Ler arquivos do fold
    images_file = fold_path / f"fold{fold_num}_images.txt"
    labels_file = fold_path / f"fold{fold_num}_labels.txt"

    if not images_file.exists() or not labels_file.exists():
        raise FileNotFoundError(f"Arquivos do fold {fold_num} não encontrados")

    # Ler listas de arquivos
    with open(images_file, "r") as f:
        image_names = [line.strip() for line in f if line.strip()]

    with open(labels_file, "r") as f:
        label_names = [line.strip() for line in f if line.strip()]

    # Criar caminhos completos
    base_path = Path(data_path)
    image_paths = [base_path / "images" / name for name in image_names]
    label_paths = [base_path / "labels" / name for name in label_names]

    return image_paths, label_paths


def load_kfold_data(data_path, test_fold):
    """
    Carrega dados usando K-fold onde:
    - test_fold: fold usado para teste
    - test_fold-1: fold usado para validação (com wrap-around)
    - demais folds: usados para treino
    """
    all_folds = [1, 2, 3, 4, 5]

    # Fold de teste
    test_images, test_labels = load_fold_files(data_path, test_fold)

    # Fold de validação (fold anterior ao teste, com wrap-around)
    val_fold = 5 if test_fold == 1 else test_fold - 1
    val_images, val_labels = load_fold_files(data_path, val_fold)

    # Folds de treino (todos os outros)
    train_folds = [f for f in all_folds if f != test_fold and f != val_fold]
    train_images, train_labels = [], []

    for fold in train_folds:
        fold_images, fold_labels = load_fold_files(data_path, fold)
        train_images.extend(fold_images)
        train_labels.extend(fold_labels)

    print(f"Fold {test_fold} como teste:")
    print(f"  Treino: folds {train_folds} ({len(train_images)} imagens)")
    print(f"  Validação: fold {val_fold} ({len(val_images)} imagens)")
    print(f"  Teste: fold {test_fold} ({len(test_images)} imagens)")

    return {
        "train": {"images": train_images, "labels": train_labels},
        "val": {"images": val_images, "labels": val_labels},
        "test": {"images": test_images, "labels": test_labels},
    }


# ========================================
# DATASET
# ========================================


class AugmentedDataset(Dataset):
    def __init__(self, image_paths, label_paths, image_size=(512, 512), augment=True):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.image_size = image_size
        self.augment = augment

        # Data augmentation
        if augment:
            import albumentations as A

            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.Rotate(limit=30, p=0.3),
                    A.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.3
                    ),
                    A.GaussianBlur(blur_limit=3, p=0.2),
                    A.GaussNoise(var_limit=(10, 50), p=0.2),
                ]
            )
        else:
            self.transform = None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Carregar imagem
        image = cv2.imread(str(self.image_paths[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size)

        # Carregar label (anotação manual em RGB)
        label = cv2.imread(str(self.label_paths[idx]))
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        label = cv2.resize(label, self.image_size, interpolation=cv2.INTER_NEAREST)
        label = rgb_to_class(label)  # Converte RGB manual para classes numéricas

        # Aplicar augmentation
        if self.transform is not None:
            transformed = self.transform(image=image, mask=label)
            image = transformed["image"]
            label = transformed["mask"]

        # Normalizar imagem
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # CHW

        return torch.FloatTensor(image), torch.LongTensor(label)


# ========================================
# MODELO
# ========================================


def create_model(num_classes=8):
    """Cria modelo U-Net com encoder mais poderoso"""
    model = smp.Unet(
        encoder_name="efficientnet-b4",  # Encoder mais poderoso
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
        activation=None,  # Sigmoid será aplicado na loss
    )
    return model


# ========================================
# TREINAMENTO
# ========================================


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0

    # Métricas
    acc_metric = Accuracy(task="multiclass", num_classes=CONFIG["num_classes"]).to(
        device
    )
    f1_metric = F1Score(
        task="multiclass", num_classes=CONFIG["num_classes"], average="macro"
    ).to(device)
    iou_metric = JaccardIndex(task="multiclass", num_classes=CONFIG["num_classes"]).to(
        device
    )

    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Métricas
        preds = torch.argmax(outputs, dim=1)
        acc = acc_metric(preds, labels)
        f1 = f1_metric(preds, labels)
        iou = iou_metric(preds, labels)

        running_loss += loss.item()

        pbar.set_postfix(
            {
                "Loss": f"{loss.item():.4f}",
                "Acc": f"{acc:.4f}",
                "F1": f"{f1:.4f}",
                "IoU": f"{iou:.4f}",
            }
        )

    return running_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0

    acc_metric = Accuracy(task="multiclass", num_classes=CONFIG["num_classes"]).to(
        device
    )
    f1_metric = F1Score(
        task="multiclass", num_classes=CONFIG["num_classes"], average="macro"
    ).to(device)
    iou_metric = JaccardIndex(task="multiclass", num_classes=CONFIG["num_classes"]).to(
        device
    )

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            preds = torch.argmax(outputs, dim=1)
            acc_metric.update(preds, labels)
            f1_metric.update(preds, labels)
            iou_metric.update(preds, labels)

            running_loss += loss.item()

    return {
        "loss": running_loss / len(dataloader),
        "acc": acc_metric.compute().item(),
        "f1": f1_metric.compute().item(),
        "iou": iou_metric.compute().item(),
    }


# ========================================
# TREINAMENTO POR FOLD
# ========================================


def train_single_fold(data_splits, fold_id):
    """Treina um fold específico"""
    # Criar pasta para modelos
    os.makedirs("modelos", exist_ok=True)

    # Criar datasets com augmentation
    train_dataset = AugmentedDataset(
        data_splits["train"]["images"],
        data_splits["train"]["labels"],
        CONFIG["image_size"],
        augment=True,  # Augmentation apenas no treino
    )
    val_dataset = AugmentedDataset(
        data_splits["val"]["images"],
        data_splits["val"]["labels"],
        CONFIG["image_size"],
        augment=False,  # Sem augmentation na validação
    )

    # Criar dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    print(
        f"Dados carregados: {len(train_dataset)} treino, {len(val_dataset)} validação"
    )

    # Criar modelo
    print("Criando modelo...")
    model = create_model(CONFIG["num_classes"]).to(CONFIG["device"])

    # Loss functions combinadas
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = smp.losses.DiceLoss(mode="multiclass")
    focal_loss = smp.losses.FocalLoss(mode="multiclass")

    def combined_loss(outputs, targets):
        ce = ce_loss(outputs, targets)
        dice = dice_loss(outputs, targets)
        focal = focal_loss(outputs, targets)
        return 0.5 * ce + 0.3 * dice + 0.2 * focal

    # Otimizador e scheduler melhorados
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

    # Early stopping
    best_iou = 0
    patience = 15
    patience_counter = 0

    print("Iniciando treinamento...")

    for epoch in range(CONFIG["epochs"]):
        print(f"Época {epoch+1}/{CONFIG['epochs']}")

        # Treinar
        train_loss = train_epoch(
            model, train_loader, combined_loss, optimizer, CONFIG["device"]
        )

        # Validar
        val_metrics = validate(model, val_loader, ce_loss, CONFIG["device"])

        # Scheduler
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}")
        print(
            f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.4f}, "
            f"F1: {val_metrics['f1']:.4f}, IoU: {val_metrics['iou']:.4f}"
        )

        # Salvar melhor modelo
        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            torch.save(model.state_dict(), f"modelos/best_model_fold_{fold_id}.pth")
            print(f"Novo melhor modelo salvo! IoU: {best_iou:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping após {patience} épocas sem melhoria")
            break

    return {"fold": fold_id, "best_iou": best_iou, "final_metrics": val_metrics}


# ========================================
# FUNÇÃO PRINCIPAL
# ========================================


def main():
    print("🚀 Iniciando K-fold Cross Validation para Segmentação Semântica")
    print(f"Device: {CONFIG['device']}")

    # Executar K-fold Cross Validation (5 folds)
    all_results = []

    for fold in range(1, 6):  # Folds 1, 2, 3, 4, 5
        print(f"\n{'='*50}")
        print(f"FOLD {fold}/5 (usando fold {fold} como teste)")
        print(f"{'='*50}")

        # Carregar dados para este fold
        data_splits = load_kfold_data(CONFIG["data_path"], test_fold=fold)

        # Treinar modelo para este fold
        fold_result = train_single_fold(data_splits, fold)
        all_results.append(fold_result)

        print(f"Fold {fold} finalizado - IoU: {fold_result['best_iou']:.4f}")

    # Calcular estatísticas finais
    ious = [r["best_iou"] for r in all_results]
    mean_iou = sum(ious) / len(ious)
    std_iou = (sum([(x - mean_iou) ** 2 for x in ious]) / len(ious)) ** 0.5

    print(f"\n{'='*50}")
    print("RESULTADOS FINAIS K-FOLD")
    print(f"{'='*50}")
    print("IoU por fold:")
    for i, iou in enumerate(ious):
        print(f"  Fold {i+1}: {iou:.4f}")
    print(f"\nResultado final: {mean_iou:.4f} ± {std_iou:.4f}")
    print(f"Melhor fold: {max(ious):.4f}")
    print(f"Pior fold: {min(ious):.4f}")

    return all_results


# ========================================
# INFERÊNCIA
# ========================================


def predict_image(model_path, image_path, output_path):
    """Faz predição em uma imagem"""
    # Carregar modelo
    model = create_model(CONFIG["num_classes"])
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Carregar imagem
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]
    image = cv2.resize(image, CONFIG["image_size"])
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = torch.FloatTensor(image).unsqueeze(0)

    # Predição
    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1).squeeze().numpy()

    # Converter para RGB e redimensionar
    pred_rgb = class_to_rgb(pred)
    pred_rgb = cv2.resize(
        pred_rgb, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST
    )

    # Salvar
    cv2.imwrite(output_path, cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR))
    print(f"Predição salva em: {output_path}")


# ========================================
# EXECUÇÃO
# ========================================

if __name__ == "__main__":
    # Para treinar:
    main()

    # Para fazer predição (descomente após treinar):
    # predict_image('modelos/best_model_fold_3.pth', 'test_image.png', 'prediction.png')
