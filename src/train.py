import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from dataset import PetropolisDataset
from model import LULCSegNet
from losses import FocalLoss
from utils import load_fold_files, compute_iou


def train_model(config):
    print("🌿 LULC-SegNet - Treinamento para Mata Atlântica (Petrópolis)")
    print(f"Device: {config.DEVICE}")

    # Carregar folds (fold1 = treino, fold2 = validação)
    train_images, train_labels = load_fold_files(config.DATA_PATH, 1)
    val_images, val_labels = load_fold_files(config.DATA_PATH, 2)

    # Datasets e DataLoaders
    train_dataset = PetropolisDataset(train_images, train_labels, config, mode="train")
    val_dataset = PetropolisDataset(val_images, val_labels, config, mode="val")
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    print(f"Treino: {len(train_dataset)} amostras")
    print(f"Validação: {len(val_dataset)} amostras")

    # Modelo
    model = LULCSegNet(config.NUM_CLASSES).to(config.DEVICE)

    # Loss, otimizador e scheduler
    criterion = FocalLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS
    )

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

            # Checa se há classes inválidas
            assert (
                masks.max() < config.NUM_CLASSES
            ), f"Classe inválida encontrada: {masks.max().item()}"

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
        all_ious = []
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(config.DEVICE), masks.to(config.DEVICE)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                ious = compute_iou(preds, masks, config.NUM_CLASSES)
                all_ious.extend(ious)

        valid_ious = [iou for iou in all_ious if not np.isnan(iou)]
        miou = np.mean(valid_ious) if valid_ious else 0.0
        scheduler.step()

        print(
            f"Época {epoch+1}: "
            f"Train Loss: {train_loss/len(train_loader):.4f}, "
            f"Val Loss: {val_loss/len(val_loader):.4f}, "
            f"mIoU: {miou:.4f}"
        )

        # Salva o melhor modelo
        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), "lulc_segnet_best.pth")
            print(f"✅ Novo melhor modelo salvo! mIoU: {best_miou:.4f}")

    print(f"\n🎯 Melhor mIoU: {best_miou:.4f}")
    return model
