from ultralytics import YOLO
import os

PATH = os.getcwd()
model = YOLO("yolo11n-cls.pt")

model.train(
    data=PATH,
    epochs=10,
    imgsz=256,
    project="runs",
    name="classify"
)
model.val(
    data=PATH,
    split="test"
)
