import cv2
import time
import numpy as np
import mediapipe as mp
from collections import deque

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

print("IRIS vs VIDEO — STEP-3 (DUAL MODE)")
print("I → IRIS | V → VIDEO | Q → Quit")

cap = cv2.VideoCapture(0)

# ===============================
# MEDIAPIPE INIT
# ===============================
base_options = python.BaseOptions(
    model_asset_path="models/face_landmarker.task"
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    num_faces=1
)

landmarker = vision.FaceLandmarker.create_from_options(options)

# ===============================
# STATE
# ===============================
mode = "IRIS"
prev_y = None
prev_dy = 0.0
jerk_window = deque(maxlen=20)

# ===============================
# JERK FUNCTION
# ===============================
def compute_jerk(y):
    global prev_y, prev_dy
    if prev_y is None:
        prev_y = y
        prev_dy = 0.0
        return 0.0
    dy = y - prev_y
    jerk = abs(dy - prev_dy)
    prev_y = y
    prev_dy = dy
    return jerk

# ===============================
# LOOP
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
    elif key == ord("q"):
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    phase = "NO DATA"
    jerk_val = 0.0
    variance = 0.0

    result = landmarker.detect(mp_image)

    if result.face_landmarks:
        nose = result.face_landmarks[0][1]

        # IRIS: smooth intent
        if mode == "IRIS":
            jerk_val = compute_jerk(nose.y)

        # VIDEO: pixel chaos amplification
        else:
            noise = np.random.normal(0, 0.003)
            jerk_val = compute_jerk(nose.y + noise)

        jerk_window.append(jerk_val)

        if len(jerk_window) > 5:
            variance = np.var(jerk_window)

        # PHASE LOGIC
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
    # OVERLAY
    # ===============================
    color = (0,255,0)
    if phase == "JITTER":
        color = (0,255,255)
    elif phase == "PRE-COLLAPSE":
        color = (0,165,255)
    elif phase == "COLLAPSE":
        color = (0,0,255)

    cv2.putText(frame, f"MODE: {mode}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

    cv2.putText(frame, f"PHASE: {phase}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.putText(frame, f"JERK: {jerk_val:.5f}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.putText(frame, f"VARIANCE: {variance:.6f}", (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # Phase bar
    bar_w = int(min(variance * 3_000_000, 300))
    cv2.rectangle(frame, (20, 180),
                  (20 + bar_w, 195),
                  color, -1)

    cv2.imshow("IRIS — Intent vs Video", frame)

cap.release()
cv2.destroyAllWindows()