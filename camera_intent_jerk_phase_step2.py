import cv2
import time
import numpy as np
import mediapipe as mp
from collections import deque

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ===============================
# INIT
# ===============================
print("IRIS STEP-2 — PREDICTIVE PHASE INTELLIGENCE")
print("Press Q to quit")

cap = cv2.VideoCapture(0)

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
# JERK + VARIANCE STATE
# ===============================
prev_y = None
prev_dy = 0.0
jerk_window = deque(maxlen=20)

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
# MAIN LOOP
# ===============================
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

    phase = "NO FACE"
    warning = ""
    jerk_val = 0.0
    variance = 0.0

    if result.face_landmarks:
        nose = result.face_landmarks[0][1]
        jerk_val = compute_jerk(nose.y)
        jerk_window.append(jerk_val)

        if len(jerk_window) > 5:
            variance = np.var(jerk_window)

        # ---- PHASE LOGIC ----
        if variance < 0.00001:
            phase = "STABLE"
        elif variance < 0.00005:
            phase = "JITTER"
        elif variance < 0.00015:
            phase = "PRE-COLLAPSE"
            warning = "⚠️ INSTABILITY RISING"
        else:
            phase = "COLLAPSE"
            warning = "❌ SYSTEM BREAKDOWN"

        h, w, _ = frame.shape
        cx, cy = int(nose.x * w), int(nose.y * h)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    # ===============================
    # OVERLAY
    # ===============================
    cv2.putText(frame, f"PHASE: {phase}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

    cv2.putText(frame, f"JERK: {jerk_val:.5f}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.putText(frame, f"VARIANCE: {variance:.6f}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    if warning:
        cv2.putText(frame, warning, (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 3)

    # Timeline bar (🔥 judge bait)
    bar_x = 20
    bar_y = 170
    bar_w = int(min(variance * 3_000_000, 300))
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + 15),
                  (0, 0, 255), -1)

    cv2.imshow("IRIS — STEP 2", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()