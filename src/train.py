import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from pathlib import Path

from dataset import PetropolisDataset
from model import LULCSegNet
from losses import FocalLoss
from utils import load_fold_files, compute_iou, print_ious

# Diretórios globais
ROOT = Path(__file__).resolve().parent.parent   # sobe da pasta src para raiz
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)


def train_model(config):
    log_path = OUTPUTS_DIR / "training_log.txt"

    # sobrescreve log no início
    with open(log_path, "w") as f:
        f.write("📓 Log de treinamento\n")

    best_overall_miou = 0.0

    for fold_num in range(1, config.N_FOLDS + 1):
        print(f"\n🔹 Treinando Fold {fold_num}/{config.N_FOLDS}")

        train_images, train_labels = load_fold_files(config.DATA_PATH, fold_num)
        val_images, val_labels = load_fold_files(
            config.DATA_PATH, fold_num + 1 if fold_num < config.N_FOLDS else 1
        )

        train_dataset = PetropolisDataset(
            train_images, train_labels, config, mode="train", patch_size=config.IMAGE_SIZE
        )
        val_dataset = PetropolisDataset(
            val_images, val_labels, config, mode="val", patch_size=config.IMAGE_SIZE
        )

        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

        model = LULCSegNet(num_classes=config.NUM_CLASSES).to(config.DEVICE)
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

        best_fold_miou = 0.0

        for epoch in range(config.EPOCHS):
            # ================= Treino =================
            model.train()
            train_loss = 0
            for images, masks in tqdm(train_loader, desc=f"Época {epoch+1}/{config.EPOCHS}"):
                images, masks = images.to(config.DEVICE), masks.to(config.DEVICE)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # ================= Validação =================
            model.eval()
            all_preds, all_targets = [], []
            val_loss = 0
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(config.DEVICE), masks.to(config.DEVICE)
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()
                    preds = torch.argmax(outputs, dim=1)
                    all_preds.append(preds.cpu())
                    all_targets.append(masks.cpu())

            all_preds = torch.cat(all_preds, dim=0)
            all_targets = torch.cat(all_targets, dim=0)

            ious, miou = compute_iou(all_preds, all_targets, config.NUM_CLASSES)
            print(
                f"\nÉpoca {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, "
                f"Val Loss={val_loss/len(val_loader):.4f}, mIoU={miou*100:.2f}%"
            )
            print_ious(ious, config.CLASS_NAMES)

            if miou > best_fold_miou:
                best_fold_miou = miou
                torch.save(model.state_dict(), MODELS_DIR / f"lulc_segnet_best_fold{fold_num}.pth")
                print(f"✅ Novo melhor modelo do fold {fold_num} salvo! mIoU={best_fold_miou*100:.2f}%")

            scheduler.step()

        # Salvar métricas no log
        with open(log_path, "a") as f:
            f.write(f"Melhor mIoU fold {fold_num}: {best_fold_miou*100:.2f}\n")

        if best_fold_miou > best_overall_miou:
            best_overall_miou = best_fold_miou

    print(f"\n🎯 Melhor mIoU geral: {best_overall_miou*100:.2f}%")
    return model
