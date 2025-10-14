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
        0: (255, 0, 0),        # Vermelho (Urbano)
        1: (38, 115, 0),       # Verde escuro (Vegetação Densa)
        2: (0, 0, 0),          # Preto (Sombra)
        3: (133, 199, 126),    # Verde claro (Vegetação Esparsa)
        4: (255, 255, 0),      # Amarelo (Agricultura)
        5: (128, 128, 128),    # Cinza (Rocha)
        6: (139, 69, 19),      # Marrom (Solo Exposto)
        7: (84, 117, 168),     # Azul (Água)
    }
    EPOCHS = 150
    LEARNING_RATE = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    N_FOLDS = 5
