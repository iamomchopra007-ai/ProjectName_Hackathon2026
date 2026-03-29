import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

print("IRIS — STEP-4: LIVE BANDWIDTH COLLAPSE")
print("I → IRIS | V → VIDEO | B/N → Bandwidth ↓/↑ | Q → Quit")

cap = cv2.VideoCapture(0)

# ===============================
# MEDIAPIPE INIT
# ===============================
base_options = python.BaseOptions(
    model_asset_path="models/face_landmarker.task"
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1
)

landmarker = vision.FaceLandmarker.create_from_options(options)

# ===============================
# STATE
# ===============================
mode = "IRIS"
bandwidth = 1.0   # 1.0 = full, 0.01 = collapse
prev_y = None
prev_dy = 0.0
jerk_window = deque(maxlen=25)

# ===============================
def compute_jerk(y):
    global prev_y, prev_dy
    if prev_y is None:
        prev_y = y
        return 0.0
    dy = y - prev_y
    jerk = abs(dy - prev_dy)
    prev_y = y
    prev_dy = dy
    return jerk

# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    key = cv2.waitKey(1) & 0xFF
    if key == ord("i"):
        mode = "IRIS"
        jerk_window.clear()
    elif key == ord("v"):
        mode = "VIDEO"
        jerk_window.clear()
    elif key == ord("b"):
        bandwidth = max(0.01, bandwidth - 0.05)
    elif key == ord("n"):
        bandwidth = min(1.0, bandwidth + 0.05)
    elif key == ord("q"):
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # VIDEO degradation
    if mode == "VIDEO":
        noise = np.random.normal(0, (1 - bandwidth) * 0.01, rgb.shape)
        rgb = np.clip(rgb + noise * 255, 0, 255).astype(np.uint8)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    jerk = 0.0
    variance = 0.0
    phase = "NO DATA"

    result = landmarker.detect(mp_image)
    if result.face_landmarks:
        nose = result.face_landmarks[0][1]
        jerk = compute_jerk(nose.y)
        jerk_window.append(jerk)

        if len(jerk_window) > 5:
            variance = np.var(jerk_window)

        if variance < 0.00001:
            phase = "STABLE"
        elif variance < 0.00005:
            phase = "JITTER"
        elif variance < 0.00015:
            phase = "PRE-COLLAPSE"
        else:
            phase = "COLLAPSE"

        h, w, _ = frame.shape
        cx, cy = int(nose.x * w), int(nose.y * h)
        cv2.circle(frame, (cx, cy), 5, (0,255,0), -1)

    # ===============================
    # UI
    # ===============================
    color = (0,255,0)
    if phase == "JITTER": color = (0,255,255)
    if phase == "PRE-COLLAPSE": color = (0,165,255)
    if phase == "COLLAPSE": color = (0,0,255)

    cv2.putText(frame, f"MODE: {mode}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

    cv2.putText(frame, f"BANDWIDTH: {bandwidth:.2f}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)

    cv2.putText(frame, f"PHASE: {phase}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.putText(frame, f"JERK: {jerk:.5f}", (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    bar = int((1 - bandwidth) * 300)
    cv2.rectangle(frame, (20, 190),
                  (20 + bar, 205), (0,0,255), -1)

    cv2.imshow("IRIS — Bandwidth Collapse Demo", frame)

cap.release()
cv2.destroyAllWindows()