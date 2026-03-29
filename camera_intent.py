import cv2
import time
import csv
import os
import numpy as np
import mediapipe as mp

from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

print("IRIS CAMERA BOOTING… (Bandwidth-Stressed Intent)")

# ==============================
# CONFIG — LIVE NETWORK
# ==============================
INTENT_KBPS = 0.08          # start stable
BITS_PER_PACKET = 32       # intent packet size
PACKET_DROP = 0.0          # simulate loss
EMA_ALPHA = 0.15           # smoothing strength

# ==============================
# Token Bucket
# ==============================
class TokenBucket:
    def __init__(self, kbps):
        self.rate = kbps * 1000
        self.capacity = self.rate
        self.tokens = self.capacity
        self.last = time.time()

    def allow(self, bits):
        now = time.time()
        dt = now - self.last
        self.last = now
        self.tokens = min(self.capacity, self.tokens + self.rate * dt)
        if bits <= self.tokens:
            self.tokens -= bits
            return True
        return False

bucket = TokenBucket(INTENT_KBPS)

# ==============================
# Logging
# ==============================
LOG_PATH = "logs/intent_camera_log.csv"
os.makedirs("logs", exist_ok=True)

log_file = open(LOG_PATH, "w", newline="")
logger = csv.writer(log_file)
logger.writerow(["step", "x", "y", "velocity", "jerk", "kbps"])

# ==============================
# MediaPipe Face
# ==============================
options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="models/face_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO
)

landmarker = vision.FaceLandmarker.create_from_options(options)

# ==============================
# Camera
# ==============================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera not accessible")

print("LIVE CONTROLS:")
print("  b / B → bandwidth down / up")
print("  d / D → packet drop up / down")
print("Press Q to quit")

# ==============================
# Intent State
# ==============================
prev_pos = None
prev_vel = 0.0
smooth_x, smooth_y = None, None
prev_time = time.time()
step = 0

# ==============================
# MAIN LOOP
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    key = cv2.waitKey(1) & 0xFF

    if key == ord('b'):
        INTENT_KBPS = max(0.01, INTENT_KBPS - 0.01)
        bucket.rate = INTENT_KBPS * 1000
        print(f"↓ Bandwidth {INTENT_KBPS:.2f} kbps")

    if key == ord('B'):
        INTENT_KBPS += 0.01
        bucket.rate = INTENT_KBPS * 1000
        print(f"↑ Bandwidth {INTENT_KBPS:.2f} kbps")

    if key == ord('d'):
        PACKET_DROP = min(0.5, PACKET_DROP + 0.05)
        print(f"↑ Drop {PACKET_DROP:.2f}")

    if key == ord('D'):
        PACKET_DROP = max(0.0, PACKET_DROP - 0.05)
        print(f"↓ Drop {PACKET_DROP:.2f}")

    if key == ord('q'):
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )

    timestamp_ms = int(time.time() * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    velocity = jerk = 0.0

    if result.face_landmarks:
        lm = result.face_landmarks[0][1]
        cx, cy = int(lm.x * w), int(lm.y * h)

        now = time.time()
        dt = max(now - prev_time, 1e-6)

        if prev_pos:
            dx = cx - prev_pos[0]
            dy = cy - prev_pos[1]
            velocity = np.sqrt(dx*dx + dy*dy) / dt
            jerk = abs(velocity - prev_vel) / dt

        # ==============================
        # BANDWIDTH GATE
        # ==============================
        send = bucket.allow(BITS_PER_PACKET) and np.random.rand() > PACKET_DROP

        if send:
            smooth_x = cx if smooth_x is None else (
                EMA_ALPHA * cx + (1 - EMA_ALPHA) * smooth_x
            )
            smooth_y = cy if smooth_y is None else (
                EMA_ALPHA * cy + (1 - EMA_ALPHA) * smooth_y
            )
        # else → HOLD previous (intent inference)

        prev_pos = (cx, cy)
        prev_vel = velocity
        prev_time = now

        logger.writerow([
            step,
            int(smooth_x),
            int(smooth_y),
            round(velocity, 3),
            round(jerk, 3),
            round(INTENT_KBPS, 3)
        ])
        log_file.flush()

        # ==============================
        # VISUAL
        # ==============================
        cv2.circle(frame, (int(smooth_x), int(smooth_y)), 6, (0,255,0), -1)

        cv2.putText(frame, "IRIS — Intent under Bandwidth Stress",
                    (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)

        cv2.putText(frame, f"KBPS: {INTENT_KBPS:.2f}",
                    (20,65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.putText(frame, f"Drop: {PACKET_DROP:.2f}",
                    (20,95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        step += 1

    cv2.imshow("IRIS — Live Intent (Bandwidth Controlled)", frame)

cap.release()
log_file.close()
cv2.destroyAllWindows()

print("✅ UPGRADE 3 COMPLETE — Intent survives bandwidth collapse")
