from ultralytics import YOLO
from pathlib import Path


MODEL_PATH = Path("runs/classify2/weights/last.pt")
IMAGE_PATH = Path(r"C:\Users\HP\Downloads\city.jpg")  # ← zmień na dowolny obraz

model = YOLO(MODEL_PATH)

results = model(IMAGE_PATH)

result = results[0]

class_names = result.names

# Najlepsza predykcja
top1_idx = result.probs.top1
top1_conf = result.probs.top1conf
top1_name = class_names[top1_idx]

print("=== WYNIK KLASYFIKACJI ===")
print(f"Obraz: {IMAGE_PATH.name}")
print(f"Klasa: {top1_name}")
print(f"Pewność: {top1_conf:.2f}")

# (OPCJONALNIE) TOP-5
print("\nTop-5 predykcji:")
for idx, conf in zip(result.probs.top5, result.probs.top5conf):
    print(f"{class_names[idx]}: {conf:.2f}")
