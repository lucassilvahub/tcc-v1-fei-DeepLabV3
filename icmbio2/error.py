import os
import numpy as np
from PIL import Image

# Caminhos das pastas
pasta1 = '../../dataset_35/labels'
out = 'output_35/20250720K4_b75_drop_Plato_flipRotColor_unet_focal_loss_RADAM'
pasta2 = f'{out}/inference'
pasta_saida = f'{out}/erro_teste'

#L = [20,32,4,12,9,17,4]
#L = [15,5,21,29,16,3]
#L = [10,14,19,2,6,23,1,7,13,18,22,24,25,26,27,28,30,31,8,11]
L = [x for x in range(1,36)]

os.makedirs(pasta_saida, exist_ok=True)

for i in L:
    nome_arquivo = f'raster{i:02}.png'
    nome_arquivo2 = f'inference_tile_raster{i:02}.png'
    caminho1 = os.path.join(pasta1, nome_arquivo)
    caminho2 = os.path.join(pasta2, nome_arquivo2)
    
    if not (os.path.exists(caminho1)):
        print(f'Imagem {nome_arquivo} nao encontrada.')
        continue
    
    if not (os.path.exists(caminho2)):
        print(f'Imagem {nome_arquivo2} nao encontrada.')
        continue

    # Abre as duas imagens e converte para RGB
    img1 = Image.open(caminho1).convert('RGB')
    img2 = Image.open(caminho2).convert('RGB')

    # Converte para arrays numpy
    arr1 = np.array(img1, dtype=np.uint8)
    arr2 = np.array(img2, dtype=np.uint8)

    if arr1.shape != arr2.shape:
        print(f'Tamanho diferente para {nome_arquivo}, pulando...')
        continue

    iguais = np.all(arr1 == arr2, axis=-1)  # shape (H, W)

    # - Branco (255) onde iguais
    # - Cor original de arr1 onde diferentes
    # usamos broadcasting de [255,255,255] no lugar de arr1 quando iguais
    cores_corretas = np.where(iguais[..., None], 255, arr1).astype(np.uint8)  # shape (H, W, 3)
    cores_erradas = np.where(iguais[..., None], 255, arr2).astype(np.uint8)  # shape (H, W, 3)

    # Salva a imagem resultante
    cores_corretas = Image.fromarray(cores_corretas, mode='RGB')
    cores_erradas = Image.fromarray(cores_erradas, mode='RGB')

    cores_corretas.save(os.path.join(pasta_saida, f'raster{i:02}_correto.png'))
    cores_erradas.save(os.path.join(pasta_saida, f'raster{i:02}_errado.png'))
    print(f'Imagem de erro salva: {nome_arquivo}')
