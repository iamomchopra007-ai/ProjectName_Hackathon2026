import cv2
import time
import math
import numpy as np

from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

# ===============================
# CONFIG
# ===============================
MODEL_PATH = "models/face_landmarker.task"
MODE = "IRIS"   # default
bandwidth = 20.0   # kbps (simulated)
last_pos = None
last_vel = 0.0
last_time = time.time()

# ===============================
# MEDIAPIPE SETUP
# ===============================
options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)

landmarker = vision.FaceLandmarker.create_from_options(options)

# ===============================
# CAMERA
# ===============================
cap = cv2.VideoCapture(0)

print("IRIS vs VIDEO — Phase Boundary Demo")
print("I → IRIS MODE | V → VIDEO MODE | Q → Quit")

# ===============================
# MAIN LOOP
# ===============================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    dt = now - last_time
    last_time = now

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # -------- VIDEO STRESS (STEP-4B)
    if MODE == "VIDEO":
        frame = cv2.GaussianBlur(frame, (21, 21), 0)
        if np.random.rand() < 0.3:
            continue
        bandwidth = 25 + np.random.randn() * 3
    else:
        bandwidth = 0.02 + abs(np.random.randn()) * 0.01

    # -------- FACE DETECTION
    result = landmarker.detect(rgb)

    jerk = 0.0

    if result.face_landmarks:
        lm = result.face_landmarks[0][1]  # nose tip
        x, y = lm.x, lm.y

        if last_pos is not None and dt > 0:
            vel = math.dist((x, y), last_pos) / dt
            jerk = (vel - last_vel) / dt
            last_vel = vel

        last_pos = (x, y)

        cx = int(x * frame.shape[1])
        cy = int(y * frame.shape[0])
        cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)

    # ===============================
    # STEP-4.2 — PHASE BOUNDARIES
    # ===============================
    if MODE == "IRIS":
        if abs(jerk) < 300:
            phase = "STABLE (Intent Preserved)"
            color = (0, 255, 0)
        elif abs(jerk) < 1500:
            phase = "UNSTABLE (Edge)"
            color = (0, 255, 255)
        else:
            phase = "COLLAPSE (Intent Fails)"
            color = (0, 0, 255)
    else:
        if bandwidth > 10:
            phase = "WASTEFUL (Pixel Heavy)"
            color = (0, 0, 255)
        else:
            phase = "TEMPORARY"
            color = (0, 255, 255)

    # ===============================
    # STEP-4.3 — INTENT EFFICIENCY
    # ===============================
    efficiency = abs(jerk) / max(bandwidth, 1e-6)

    # ===============================
    # OVERLAY
    # ===============================
    cv2.putText(frame, f"MODE: {MODE}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.putText(frame, f"JERK: {jerk:.2f}",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.putText(frame, f"PHASE: {phase}",
                (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.putText(frame, f"BW (kbps): {bandwidth:.3f}",
                (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    cv2.putText(frame, f"Intent/kbps: {efficiency:.2f}",
                (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

    cv2.imshow("IRIS — Intent vs Pixels", frame)

    # ===============================
    # CONTROLS
    # ===============================
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('i'):
        MODE = "IRIS"
    elif key == ord('v'):
        MODE = "VIDEO"

cap.release()
cv2.destroyAllWindows()