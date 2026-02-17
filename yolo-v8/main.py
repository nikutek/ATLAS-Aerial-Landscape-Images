from ultralytics import YOLO
import os

PATH = os.getcwd()
model = YOLO("yolo11n-cls.pt")

model.train(
    data=PATH,
    epochs=25,
    imgsz=256,
    project="runs",
    name="classify",

    # AUGMENTACJA
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=15,
    translate=0.1,
    scale=0.5,
    flipud=0.5,
    fliplr=0.5,

    patience=10,  # zatrzymaj jeśli brak poprawy przez 10 epok
)

model.val(
    data=PATH,
    split="test"
)