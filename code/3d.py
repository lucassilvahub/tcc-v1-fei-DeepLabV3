
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import tifffile as tiff  

image_folder = '../../dataset_35/images'
image_extensions = ('.tif', '.jpg', '.jpeg', '.png')
image_size = (2048, 2048)  

def load_rgb_pixels(folder, num_pixels_per_image=1000):
    pixels = []
    for filename in sorted(os.listdir(folder))[:35]:
        if filename.lower().endswith(image_extensions):
            path = os.path.join(folder, filename)
            if filename.lower().endswith('.tif'):
                img = tiff.imread(path)
            else:
                img = cv2.imread(path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if img is None or img.shape[-1] < 3:
                continue

            img = img[..., :3]
            img = img.astype(np.float32) / 255.0

            h, w, _ = img.shape
            pixels_flat = img.reshape(-1, 3)
            idx = np.random.choice(pixels_flat.shape[0], size=min(num_pixels_per_image, pixels_flat.shape[0]), replace=False)
            sampled = pixels_flat[idx]
            pixels.append(sampled)

            print(f"Imagem {filename} processada com {sampled.shape[0]} pixels.")
    
    return np.concatenate(pixels, axis=0) if pixels else np.array([])


rgb_pixels = load_rgb_pixels(image_folder, num_pixels_per_image=512*512)

if rgb_pixels.size == 0:
    raise RuntimeError("Nenhum pixel carregado.")

print(f"Total de pixels: {rgb_pixels.shape[0]}")

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
r, g, b = rgb_pixels[:, 0], rgb_pixels[:, 1], rgb_pixels[:, 2]
ax.scatter(r, g, b, c=rgb_pixels, marker='o', s=0.5)  
ax.set_xlabel('Red')
ax.set_ylabel('Green')
ax.set_zlabel('Blue')
ax.set_title('Espaco de Cores RGB dos Pixels')
plt.tight_layout()
plt.show()
