# Adicione estas linhas LOGO APÓS os imports:
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from torchvision import models

from dataset import PetropolisPatchDataset
from losses import FocalLoss
from utils import load_fold_files, compute_iou, print_ious

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)


def get_model(num_classes):
    model = models.segmentation.deeplabv3_resnet50(weights="DEFAULT")
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
    return model


def train_model(config):
    log_path = OUTPUTS_DIR / "training_log.txt"
    with open(log_path, "w") as f:
        f.write("Log de treinamento\n")

    best_overall_miou = 0.0

    for fold_num in range(1, config.N_FOLDS + 1):
        print(f"\nTreinando Fold {fold_num}/{config.N_FOLDS}")

        train_imgs, train_lbls = load_fold_files(config.DATA_PATH, fold_num)
        val_imgs, val_lbls = load_fold_files(
            config.DATA_PATH, fold_num + 1 if fold_num < config.N_FOLDS else 1
        )

        train_set = PetropolisPatchDataset(train_imgs, train_lbls, config, mode="train")
        val_set = PetropolisPatchDataset(val_imgs, val_lbls, config, mode="val")

        # ===== ADICIONE OS PARÂMETROS AQUI =====
        train_loader = DataLoader(
            train_set,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=8,  # <- NOVO
            pin_memory=True,  # <- NOVO
            prefetch_factor=3,  # <- NOVO
            persistent_workers=True,  # <- NOVO
        )
        val_loader = DataLoader(
            val_set,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=8,  # <- NOVO
            pin_memory=True,  # <- NOVO
            prefetch_factor=3,  # <- NOVO
            persistent_workers=True,  # <- NOVO
        )

        model = get_model(config.NUM_CLASSES).to(config.DEVICE)
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
        )

        scaler = torch.amp.GradScaler('cuda')
        best_fold_miou, patience_counter, patience = 0.0, 0, 15

        for epoch in range(1, config.EPOCHS + 1):
            # ===== Treino =====
            model.train()
            train_loss = 0
            for imgs, masks in tqdm(
                train_loader, desc=f"Época {epoch}/{config.EPOCHS}"
            ):
                imgs, masks = imgs.to(config.DEVICE), masks.to(config.DEVICE)
                optimizer.zero_grad()
                with torch.amp.autocast("cuda"):  # <- CORRIGIDO
                    outputs = model(imgs)["out"]
                    loss = criterion(outputs, masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()

            # ===== Validação =====
            model.eval()
            preds_all, targets_all, val_loss = [], [], 0
            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs, masks = imgs.to(config.DEVICE), masks.to(config.DEVICE)
                    with torch.amp.autocast("cuda"):  # <- CORRIGIDO
                        outputs = model(imgs)["out"]
                        loss = criterion(outputs, masks)
                    val_loss += loss.item()
                    preds_all.append(torch.argmax(outputs, dim=1).cpu())
                    targets_all.append(masks.cpu())

            preds_all = torch.cat(preds_all, dim=0)
            targets_all = torch.cat(targets_all, dim=0)
            ious, miou = compute_iou(preds_all, targets_all, config.NUM_CLASSES)

            print(
                f"\nÉpoca {epoch}: Train={train_loss/len(train_loader):.4f}, "
                f"Val={val_loss/len(val_loader):.4f}, mIoU={miou*100:.2f}%"
            )
            print_ious(ious, config.CLASS_NAMES)

            # Early stopping + checkpoint
            if miou > best_fold_miou:
                best_fold_miou = miou
                patience_counter = 0
                torch.save(
                    model.state_dict(),
                    MODELS_DIR / f"deeplabv3_best_fold{fold_num}.pth",
                )
                print(f"Novo melhor modelo fold {fold_num} salvo! mIoU={miou*100:.2f}%")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping ativado")
                    break

            old_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(miou)
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr < old_lr:
                print(f"Taxa de aprendizado reduzida: {old_lr:.6f} -> {new_lr:.6f}")

        with open(log_path, "a") as f:
            f.write(f"Melhor mIoU fold {fold_num}: {best_fold_miou*100:.2f}\n")

        best_overall_miou = max(best_overall_miou, best_fold_miou)

    print(f"\nMelhor mIoU geral: {best_overall_miou*100:.2f}%")
