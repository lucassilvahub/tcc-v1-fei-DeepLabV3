from config import Config
from train import train_model

if __name__ == "__main__":
    config = Config()
    print("\n===== Iniciando Treinamento Cross-Validation =====")
    train_model(config)
