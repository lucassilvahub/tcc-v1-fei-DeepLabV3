import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from dataset import PetropolisPatchDataset
from model import LULCSegNet
from losses import FocalLoss
from utils import compute_iou, load_fold_files

def train_model(config, fold_num):
    """
    Treina o modelo usando os paths de um fold específico (1 a 5).
    """
    print(f"\n🌿 LULC-SegNet - Treinamento Fold {fold_num}")
    print(f"Device: {config.DEVICE}")

    # Carrega imagens e labels do fold
    train_images, train_labels = load_fold_files(config.DATA_PATH, fold_num)
    val_images, val_labels = load_fold_files(config.DATA_PATH, fold_num+1 if fold_num < 5 else 1)

    print(f"Treino imagens: {len(train_images)}, Val imagens: {len(val_images)}")

    # Dataset e DataLoader
    train_dataset = PetropolisPatchDataset(train_images, train_labels, config, patch_size=config.PATCH_SIZE, stride=config.STRIDE, mode="train")
    val_dataset = PetropolisPatchDataset(val_images, val_labels, config, patch_size=config.PATCH_SIZE, stride=config.STRIDE, mode="val")
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Modelo
    model = LULCSegNet(config.NUM_CLASSES, pretrained=True).to(config.DEVICE)

    # Loss, otimizador e scheduler
    criterion = FocalLoss(alpha=None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    best_miou = 0.0

    for epoch in range(config.EPOCHS):
        # =======================
        # Treinamento
        # =======================
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Época {epoch+1}/{config.EPOCHS}")
        for images, masks in pbar:
            images, masks = images.to(config.DEVICE), masks.to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        # =======================
        # Validação
        # =======================
        model.eval()
        val_loss = 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(config.DEVICE), masks.to(config.DEVICE)
                outputs = model(images)
                val_loss += criterion(outputs, masks).item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu())
                all_targets.append(masks.cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        class_ious = compute_iou(all_preds, all_targets, config.NUM_CLASSES)
        valid_ious = [iou for iou in class_ious if not np.isnan(iou)]
        miou = np.mean(valid_ious) if valid_ious else 0.0

        # Logging detalhado
        print(f"\nEpoch {epoch+1}: Train Loss {train_loss/len(train_loader):.4f}, Val Loss {val_loss/len(val_loader):.4f}, mIoU {miou*100:.2f}%")
        print("📊 IoU por classe:")
        for idx, iou in enumerate(class_ious):
            print(f"  {config.CLASS_NAMES[idx]:<15}: {iou*100 if not np.isnan(iou) else 0.0:6.2f}%")

        scheduler.step()

        # Salva melhor modelo do fold
        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), f"../checkpoints/lulc_segnet_best_fold{fold_num}.pth")
            print(f"✅ Novo melhor modelo salvo! mIoU: {best_miou*100:.2f}%")

    print(f"\n🎯 Melhor mIoU Fold {fold_num}: {best_miou*100:.2f}%")
