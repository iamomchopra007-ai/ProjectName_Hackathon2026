import cv2
import time
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ===============================
# INIT
# ===============================
print("IRIS STEP-1 — JERK + PHASE + TIMELINE")
print("Press Q to quit")

cap = cv2.VideoCapture(0)

base_options = python.BaseOptions(
    model_asset_path="models/face_landmarker.task"
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=False,
    num_faces=1
)

landmarker = vision.FaceLandmarker.create_from_options(options)

# ===============================
# JERK STATE
# ===============================
prev_y = None
prev_dy = None

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

    # ✅ CRITICAL FIX — wrap numpy into mp.Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(mp_image)

    phase = "NO FACE"
    jerk_val = 0.0

    if result.face_landmarks:
        nose = result.face_landmarks[0][1]
        jerk_val = compute_jerk(nose.y)

        if jerk_val < 0.002:
            phase = "STABLE"
        elif jerk_val < 0.01:
            phase = "JITTER"
        else:
            phase = "COLLAPSE"

        h, w, _ = frame.shape
        cx, cy = int(nose.x * w), int(nose.y * h)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    cv2.putText(
        frame,
        f"PHASE: {phase} | JERK: {jerk_val:.4f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2
    )

    cv2.imshow("IRIS — STEP 1", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()