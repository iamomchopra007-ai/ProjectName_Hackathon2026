import cv2
import mediapipe as mp
import numpy as np
import time

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions

print("IRIS STEP-3 — IRIS vs VIDEO (INEVITABILITY)")
print("I → IRIS MODE | V → VIDEO MODE | Q → Quit")

# ------------------------------
# Model
# ------------------------------
MODEL_PATH = "models/face_landmarker.task"

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    num_faces=1
)

landmarker = FaceLandmarker.create_from_options(options)

# ------------------------------
# Camera
# ------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera not accessible")

# ------------------------------
# State
# ------------------------------
mode = "IRIS"
last_frame_time = time.time()
bandwidth_kbps = 0.05   # simulated stress

canvas_w, canvas_h = 640, 480

# ------------------------------
# Loop
# ------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()

    # Simulated bandwidth choke for VIDEO
    if mode == "VIDEO":
        if now - last_frame_time < 0.15:
            continue
        last_frame_time = now

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --------------------------
    # VIDEO MODE
    # --------------------------
    if mode == "VIDEO":
        noisy = cv2.GaussianBlur(frame, (15, 15), 0)
        cv2.putText(
            noisy,
            "VIDEO MODE — COLLAPSING",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )
        cv2.imshow("IRIS DEMO", noisy)

    # --------------------------
    # IRIS MODE
    # --------------------------
    else:
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = landmarker.detect(mp_image)
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        if result.face_landmarks:
            for lm in result.face_landmarks[0]:
                x = int(lm.x * canvas_w)
                y = int(lm.y * canvas_h)
                cv2.circle(canvas, (x, y), 2, (0, 255, 255), -1)

        cv2.putText(
            canvas,
            f"IRIS MODE — Presence @ {bandwidth_kbps:.2f} kbps",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.imshow("IRIS DEMO", canvas)

    # --------------------------
    # Controls
    # --------------------------
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('i'):
        mode = "IRIS"
    elif key == ord('v'):
        mode = "VIDEO"

# ------------------------------
# Cleanup
# ------------------------------
cap.release()
cv2.destroyAllWindows()
landmarker.close()