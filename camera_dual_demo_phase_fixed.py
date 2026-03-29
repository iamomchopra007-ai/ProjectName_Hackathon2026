import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

# ===============================
# CONFIG
# ===============================
MODEL_PATH = "models/face_landmarker.task"
MODE = "IRIS"   # IRIS or VIDEO

JERK_LOW = 2.0
JERK_HIGH = 8.0

# ===============================
# INIT MEDIAPIPE
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
prev_pos = None
prev_vel = 0.0
prev_time = time.time()

print("IRIS vs VIDEO — Phase Boundary Demo (FIXED)")
print("I → IRIS MODE | V → VIDEO MODE | Q → Quit")

# ===============================
# LOOP
# ===============================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('i'):
        MODE = "IRIS"
    elif key == ord('v'):
        MODE = "VIDEO"

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ✅ CORRECT IMAGE CREATION
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = landmarker.detect(mp_image)

    jerk = 0.0
    phase = "STABLE"

    if result.face_landmarks:
        lm = result.face_landmarks[0][1]  # nose
        pos = np.array([lm.x, lm.y])

        now = time.time()
        dt = max(now - prev_time, 1e-6)

        if prev_pos is not None:
            vel = np.linalg.norm((pos - prev_pos) / dt)
            jerk = abs((vel - prev_vel) / dt)
            prev_vel = vel
        else:
            vel = 0.0

        prev_pos = pos
        prev_time = now

        if jerk < JERK_LOW:
            phase = "STABLE"
        elif jerk < JERK_HIGH:
            phase = "UNSTABLE"
        else:
            phase = "COLLAPSE"

        h, w, _ = frame.shape
        cx, cy = int(pos[0] * w), int(pos[1] * h)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    # ===============================
    # OVERLAY
    # ===============================
    color = (0, 255, 0) if phase == "STABLE" else (0, 255, 255) if phase == "UNSTABLE" else (0, 0, 255)

    cv2.putText(frame, f"MODE: {MODE}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.putText(frame, f"JERK: {jerk:.2f}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.putText(frame, f"PHASE: {phase}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("IRIS — Intent vs Pixels", frame)

# ===============================
# CLEANUP
# ===============================
cap.release()
cv2.destroyAllWindows()
landmarker.close()