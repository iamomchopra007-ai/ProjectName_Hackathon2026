import cv2
import time
import numpy as np

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    ImageFrame
)

# =============================
# CONFIG
# =============================
MODEL_PATH = "models/face_landmarker.task"

JERK_LOW = 15.0
JERK_HIGH = 45.0

IRIS_MODE = True
VIDEO_NOISE = False

# =============================
# LANDMARKER
# =============================
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    num_faces=1
)

landmarker = FaceLandmarker.create_from_options(options)

# =============================
# CAMERA
# =============================
cap = cv2.VideoCapture(0)

print("\nIRIS FINAL DEMO")
print("I → IRIS MODE | V → VIDEO MODE | Q → QUIT\n")

# =============================
# STATE
# =============================
prev_pos = None
prev_vel = None
prev_acc = None
prev_time = time.time()
intent_pos = None

# =============================
# LOOP
# =============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ✅ ONLY VALID OBJECT IN YOUR MEDIAPIPE VERSION
    mp_image = ImageFrame(
        image_format=ImageFrame.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(mp_image)

    jerk = 0.0
    phase = "NO FACE"
    color = (180, 180, 180)

    if result.face_landmarks:
        nose = result.face_landmarks[0][1]
        h, w, _ = frame.shape
        raw_pos = np.array([nose.x * w, nose.y * h], dtype=np.float32)

        if intent_pos is None:
            intent_pos = raw_pos
        else:
            if IRIS_MODE:
                intent_pos = 0.85 * intent_pos + 0.15 * raw_pos
            else:
                intent_pos = raw_pos

        if VIDEO_NOISE:
            intent_pos += np.random.normal(0, 12, size=2)

        now = time.time()
        dt = max(now - prev_time, 1e-3)

        if prev_pos is not None:
            vel = (intent_pos - prev_pos) / dt
            if prev_vel is not None:
                acc = (vel - prev_vel) / dt
                if prev_acc is not None:
                    jerk = np.linalg.norm((acc - prev_acc) / dt)
                prev_acc = acc
            prev_vel = vel

        prev_pos = intent_pos.copy()
        prev_time = now

        if jerk < JERK_LOW:
            phase = "STABILITY PLATEAU"
            color = (0, 255, 0)
        elif jerk < JERK_HIGH:
            phase = "INSTABILITY REGION"
            color = (0, 255, 255)
        else:
            phase = "COLLAPSE REGION"
            color = (0, 0, 255)

        cv2.circle(frame, tuple(intent_pos.astype(int)), 6, color, -1)

    # =============================
    # OVERLAY
    # =============================
    cv2.putText(frame, f"JERK: {jerk:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.putText(frame, f"PHASE: {phase}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    mode_text = "IRIS (Intent Persists)" if IRIS_MODE else "VIDEO (Pixels Collapse)"
    mode_color = (0, 255, 0) if IRIS_MODE else (0, 0, 255)

    cv2.putText(frame, f"MODE: {mode_text}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)

    cv2.putText(frame,
                "Pixels fail. Intent survives.",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("IRIS — Intent Phase Diagram", frame)

    # =============================
    # CONTROLS
    # =============================
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('v'):
        IRIS_MODE = False
        VIDEO_NOISE = True
        print("→ VIDEO MODE (collapse)")
    elif key == ord('i'):
        IRIS_MODE = True
        VIDEO_NOISE = False
        print("→ IRIS MODE (stability)")

# =============================
# CLEANUP
# =============================
cap.release()
cv2.destroyAllWindows()
