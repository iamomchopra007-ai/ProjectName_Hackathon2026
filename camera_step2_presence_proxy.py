import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions

print("IRIS STEP-2 — RECEIVER-ONLY PRESENCE PROXY")
print("Press Q to quit")

# ------------------------------
# Load Model
# ------------------------------
MODEL_PATH = "models/face_landmarker.task"

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    num_faces=1
)

landmarker = FaceLandmarker.create_from_options(options)

# ------------------------------
# Camera (INPUT ONLY, NOT SHOWN)
# ------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera not accessible")

# ------------------------------
# Output Canvas (Receiver View)
# ------------------------------
canvas_w, canvas_h = 640, 480

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(mp_image)

    # Receiver-only canvas
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    if result.face_landmarks:
        face = result.face_landmarks[0]

        for lm in face:
            x = int(lm.x * canvas_w)
            y = int(lm.y * canvas_h)
            cv2.circle(canvas, (x, y), 2, (0, 255, 255), -1)

    cv2.putText(
        canvas,
        "IRIS — Presence (No Video)",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow("IRIS — RECEIVER", canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ------------------------------
# Cleanup
# ------------------------------
cap.release()
cv2.destroyAllWindows()
landmarker.close()