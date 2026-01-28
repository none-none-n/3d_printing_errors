import os
import numpy as np
from PIL import Image

def grayscale_to_rgb_pil(img_path):
    img = Image.open(img_path).convert('RGB')
    # Центральный квадратный кроп
    w, h = img.size
    min_side = min(w, h)
    left = (w - min_side) // 2
    top = (h - min_side) // 2
    right = left + min_side
    bottom = top + min_side
    img_cropped = img.crop((left, top, right, bottom))  # квадрат

    # Конвертация в grayscale
    img_np = np.array(img_cropped, dtype=np.float32)
    gray = 0.299 * img_np[:, :, 0] + 0.587 * img_np[:, :, 1] + 0.114 * img_np[:, :, 2]
    gray = np.expand_dims(gray, axis=-1)
    rgb_like = np.repeat(gray, 3, axis=-1).astype(np.uint8)
    return Image.fromarray(rgb_like)

def preprocess_single_folder(folder_path):
    supported_ext = ('.png', '.jpg', '.jpeg')
    image_files = [
        f for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(supported_ext)
    ]
    print(f"Найдено {len(image_files)} изображений в {folder_path}")
    for fname in image_files:
        img_path = os.path.join(folder_path, fname)
        try:
            processed_img = grayscale_to_rgb_pil(img_path)
            processed_img.save(img_path)  # перезапись
            print(f"Обработано: {fname}")
        except Exception as e:
            print(f"Ошибка при обработке {fname}: {e}")

if __name__ == '__main__':
    folder_to_process = r"C:\Users\beloz\PycharmProjects\diplom\archive (1)\no_defected"
    preprocess_single_folder(folder_to_process)
    print("Предобработка завершена.")
