import torch
import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm

from config import Config
from predict import load_models, predict_full_image
from utils import save_colored_mask

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT / "outputs" / "all_predictions"
OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)

config = Config()

if __name__ == "__main__":
    models = load_models()
    if not models:
        print("Nenhum modelo carregado! Treine primeiro.")
        exit()

    data_dir = ROOT / "data" / "images"
    
    # Listar todos os rasters
    raster_files = sorted(data_dir.glob("raster*.tif"))
    
    print(f"Encontradas {len(raster_files)} imagens para segmentar\n")
    
    for raster_path in tqdm(raster_files, desc="Segmentando imagens"):
        raster_name = raster_path.stem
        
        # Segmentar
        mask = predict_full_image(models, raster_path, patch_size=config.IMAGE_SIZE)
        
        # Salvar colorida
        output_png = OUTPUTS_DIR / f"{raster_name}_segmented.png"
        save_colored_mask(mask, config.CLASS_COLORS, str(output_png))
        
        # Salvar matriz de classes
        output_npy = OUTPUTS_DIR / f"{raster_name}_classes.npy"
        np.save(output_npy, mask)
    
    print(f"\nTodas as segmentações salvas em: {OUTPUTS_DIR}")