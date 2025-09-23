from train import train_model


class Config:
    """
    Configuração do projeto.
    """

    DATA_PATH = "../data"  # caminho para a pasta de dados
    IMAGE_SIZE = 256
    BATCH_SIZE = 8
    NUM_CLASSES = 8
    CLASS_NAMES = [
        "Mata Nativa",
        "Vegetação Densa",
        "Ocupação Urbana",
        "Solo Exposto",
        "Corpos d'Água",
        "Agricultura",
        "Regeneração",
        "Sombra",
    ]
    CLASS_COLORS = {
        0: (0, 100, 0),
        1: (0, 255, 0),
        2: (128, 128, 128),
        3: (160, 82, 45),
        4: (0, 0, 255),
        5: (255, 255, 0),
        6: (173, 255, 47),
        7: (0, 0, 0),
    }
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"


if __name__ == "__main__":
    config = Config()
    train_model(config)
