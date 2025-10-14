import torch
import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm

from config import Config
from predict import load_models, predict_full_image
from utils import save_colored_mask

# Caminhos principais
ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT / "outputs" / "all_predictions"
OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)

config = Config()

# Compatibilidade entre versões do PyTorch
if hasattr(torch, "amp"):
    autocast = torch.amp.autocast
else:
    autocast = torch.cuda.amp.autocast

if __name__ == "__main__":
    # Carregar modelos
    models = load_models()
    if not models:
        print("Nenhum modelo carregado! Treine primeiro.")
        exit()

    data_dir = ROOT / "data" / "images"

    # Listar rasters
    raster_files = sorted(data_dir.glob("raster*.tif"))

    print(f"Encontradas {len(raster_files)} imagens para segmentar\n")

    for raster_path in tqdm(raster_files, desc="Segmentando imagens"):
        raster_name = raster_path.stem

        with torch.no_grad(), autocast("cuda"):
            mask = predict_full_image(models, raster_path, patch_size=config.IMAGE_SIZE)

        # Salvar máscara colorida e matriz de classes
        output_png = OUTPUTS_DIR / f"{raster_name}_segmented.png"
        output_npy = OUTPUTS_DIR / f"{raster_name}_classes.npy"
        save_colored_mask(mask, config.CLASS_COLORS, str(output_png))
        np.save(output_npy, mask)

    print(f"\nTodas as segmentações salvas em: {OUTPUTS_DIR}")
