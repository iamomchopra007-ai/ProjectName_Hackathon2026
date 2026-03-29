import cv2
import time
import numpy as np
import mediapipe as mp

from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions

# =========================
# INIT
# =========================
print("IRIS vs VIDEO — Dual-Mode Demo")
print("I → IRIS MODE | V → VIDEO MODE | Q → Quit")

MODE = "IRIS"   # default

MODEL_PATH = "models/face_landmarker.task"

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    num_faces=1
)

landmarker = FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera not available")

# =========================
# STATE
# =========================
prev_pos = None
prev_vel = 0.0
prev_time = time.time()

# =========================
# HELPERS
# =========================
def compute_jerk(pos, prev_pos, prev_vel, dt):
    vel = np.linalg.norm(pos - prev_pos) / dt
    jerk = abs(vel - prev_vel) / dt
    return jerk, vel

def phase(j):
    if j < 300:
        return "STABLE", (0,255,0)
    elif j < 1500:
        return "UNSTABLE", (0,255,255)
    else:
        return "AGGRESSIVE", (0,0,255)

# =========================
# LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    now = time.time()
    dt = max(now - prev_time, 1e-6)
    jerk = 0.0

    # ================= IRIS MODE =================
    if MODE == "IRIS":
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = landmarker.detect(mp_image)

        if result.face_landmarks:
            nose = result.face_landmarks[0][1]
            pos = np.array([nose.x, nose.y])

            if prev_pos is not None:
                jerk, prev_vel = compute_jerk(pos, prev_pos, prev_vel, dt)

            prev_pos = pos
            prev_time = now

            cx, cy = int(nose.x * w), int(nose.y * h)
            cv2.circle(frame, (cx, cy), 6, (0,255,0), -1)

        bandwidth = 0.01  # kbps (symbolic intent stream)

    # ================= VIDEO MODE =================
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.GaussianBlur(gray, (21,21), 0)

        motion = np.sum(diff) / (w*h)
        jerk = motion / 1000.0

        bandwidth = (w*h*3*30)/1e6  # approx Mbps

    # ================= HUD =================
    state, color = phase(jerk)

    cv2.putText(frame, f"MODE: {MODE}", (20,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.putText(frame, f"JERK: {jerk:.2f}", (20,65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.putText(frame, f"PHASE: {state}", (20,100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.putText(frame, f"BW: {bandwidth:.3f}", (20,135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    cv2.imshow("IRIS — Intent vs Video", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('i'):
        MODE = "IRIS"
        prev_pos, prev_vel = None, 0.0
    elif key == ord('v'):
        MODE = "VIDEO"
        prev_pos, prev_vel = None, 0.0

cap.release()
cv2.destroyAllWindows()
