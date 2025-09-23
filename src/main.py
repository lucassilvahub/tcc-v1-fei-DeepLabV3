from config import Config
from train import train_model

config = Config()

for fold_num in range(1, 6):
    print(f"\n===== Treinando Fold {fold_num} =====")
    train_model(config, fold_num)
