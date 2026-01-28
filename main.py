import numpy as np
from PIL import Image
from ultralytics import YOLO


def preprocess_for_model(img_path):
    img = Image.open(img_path).convert('RGB')
    img_np = np.array(img, dtype=np.float32)
    gray = 0.299 * img_np[:, :, 0] + 0.587 * img_np[:, :, 1] + 0.114 * img_np[:, :, 2]
    gray = np.expand_dims(gray, axis=-1)
    rgb_like = np.repeat(gray, 3, axis=-1).astype(np.uint8)
    return Image.fromarray(rgb_like)



model = YOLO(r"runs\classify\3d_print_defect_cls_2243\weights/best.pt")

processed_img = preprocess_for_model(r"C:\Users\beloz\PycharmProjects\diplom\test\defect.jpg")
results = model(processed_img)

probs = results[0].probs
predicted_class = probs.top1
confidence = probs.top1conf.item()

print("Predicted:", model.names[predicted_class])
print("Confidence:", confidence)

#processed_img.save(r"processed_debug_yolo_input.jpg")
