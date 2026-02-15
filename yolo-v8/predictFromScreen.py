from ultralytics import YOLO
import mss
import numpy as np
import cv2
from pathlib import Path
import time


MODEL_PATH = Path("runs/classify6/weights/best.pt")
CAPTURE_SIZE = 512
FPS_DELAY = 0.5


model = YOLO(MODEL_PATH)


with mss.mss() as sct:
    monitor = sct.monitors[1]  # główny monitor

    screen_w = monitor["width"]
    screen_h = monitor["height"]

    # LEWA POŁOWA EKRANU
    left_half_width = screen_w // 2

    # środek lewej połowy
    center_x = left_half_width // 2
    center_y = screen_h // 2

    region = {
        "top": center_y - CAPTURE_SIZE // 2,
        "left": center_x - CAPTURE_SIZE // 2,
        "width": CAPTURE_SIZE,
        "height": CAPTURE_SIZE,
    }

    print("🚀 Live prediction (Google Maps on LEFT side)")
    print("ESC to quit")
    frame_id = 0

    while True:
        frame_id += 1

        screenshot = sct.grab(region)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        results = model(img, verbose=False)
        r = results[0]

        top1 = int(r.probs.top1)
        conf = float(r.probs.top1conf)
        class_name = r.names[top1]

        # PRINT DO TERMINALA
        print(f"[{frame_id}] {class_name} ({conf:.3f})")

        label = f"{class_name} ({conf:.2f})"
        cv2.putText(
            img,
            label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Aerial Landscape Prediction", img)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        time.sleep(FPS_DELAY)

cv2.destroyAllWindows()
