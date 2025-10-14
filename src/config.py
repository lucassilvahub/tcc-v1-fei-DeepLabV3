import torch


class Config:
    DATA_PATH = r"C:\Users\Lucas\Documents\College\TCC\tcc-v1-fei-DeepLabV3\data"
    IMAGE_SIZE = 256
    PATCH_SIZE = 256
    STRIDE = 128
    BATCH_SIZE = 32
    NUM_CLASSES = 8
    CLASS_NAMES = [
        "Urbano",              # 0
        "Vegetação Densa",     # 1
        "Sombra",              # 2
        "Vegetação Esparsa",   # 3
        "Agricultura",         # 4
        "Rocha",               # 5
        "Solo Exposto",        # 6
        "Água",                # 7
    ]
    CLASS_COLORS = {
        0: (0, 100, 0),          # Verde escuro (não Vermelho!)
        1: (0, 255, 0),          # Verde
        2: (128, 128, 128),      # Cinza (não Preto!)
        3: (160, 82, 45),        # Marrom
        4: (0, 0, 255),          # Azul
        5: (255, 255, 0),        # Amarelo
        6: (173, 255, 47),       # Verde claro
        7: (0, 0, 0),            # Preto
    }
    EPOCHS = 150
    LEARNING_RATE = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    N_FOLDS = 5
