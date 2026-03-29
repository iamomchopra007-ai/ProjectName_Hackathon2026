import cv2
import time
import numpy as np
from collections import deque
import mediapipe as mp

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions

MODEL_PATH = "models/face_landmarker.task"

# -----------------------------
# STATE
# -----------------------------
prev_pos = None
prev_vel = 0.0
prev_t = None
jerk_buf = deque(maxlen=5)
mode = "IRIS"

# -----------------------------
# MEDIAPIPE LANDMARKER (IMAGE MODE)
# -----------------------------
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    num_faces=1
)

landmarker = FaceLandmarker.create_from_options(options)

# -----------------------------
# CAMERA
# -----------------------------
cap = cv2.VideoCapture(0)

print("\nIRIS FINAL DEMO")
print("I → IRIS MODE | V → VIDEO MODE | Q → QUIT\n")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    now = time.time()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ✅ REQUIRED MediaPipe Image wrapper
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(mp_image)

    if result.face_landmarks:
        nose = result.face_landmarks[0][1]
        h, w, _ = frame.shape
        pos = np.array([nose.x * w, nose.y * h])

        if prev_pos is not None:
            dt = now - prev_t
            if dt > 0:
                vel = np.linalg.norm(pos - prev_pos) / dt
                acc = (vel - prev_vel) / dt
                jerk = acc / dt

                jerk_buf.append(jerk)
                j = np.mean(jerk_buf)

                if j < 5:
                    phase = "STABLE"
                    color = (0, 255, 0)
                elif j < 20:
                    phase = "MICRO"
                    color = (0, 255, 255)
                else:
                    phase = "AGGRESSIVE"
                    color = (0, 0, 255)

                cv2.putText(frame, f"MODE: {mode}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(frame, f"JERK: {j:.2f}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cv2.putText(frame, f"PHASE: {phase}", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                prev_vel = vel

        prev_pos = pos
        prev_t = now

    cv2.imshow("IRIS — Intent > Pixels", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("i"):
        mode = "IRIS"
    elif key == ord("v"):
        mode = "VIDEO"

cap.release()
cv2.destroyAllWindows()
