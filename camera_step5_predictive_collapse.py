import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

print("IRIS — STEP-5: PREDICTIVE COLLAPSE")
print("I → IRIS | V → VIDEO | B/N → BW ↓/↑ | Q → Quit")

cap = cv2.VideoCapture(0)

# ===============================
# MEDIAPIPE
# ===============================
base = python.BaseOptions(model_asset_path="models/face_landmarker.task")
opts = vision.FaceLandmarkerOptions(base_options=base, num_faces=1)
landmarker = vision.FaceLandmarker.create_from_options(opts)

# ===============================
# STATE
# ===============================
mode = "IRIS"
bandwidth = 1.0
prev_y, prev_dy = None, 0.0
jerk_hist = deque(maxlen=30)
var_hist = deque(maxlen=15)

# ===============================
def jerk(y):
    global prev_y, prev_dy
    if prev_y is None:
        prev_y = y
        return 0.0
    dy = y - prev_y
    j = abs(dy - prev_dy)
    prev_y, prev_dy = y, dy
    return j

# ===============================
while True:
    ok, frame = cap.read()
    if not ok:
        break

    key = cv2.waitKey(1) & 0xFF
    if key == ord("i"):
        mode = "IRIS"
        jerk_hist.clear()
        var_hist.clear()
    elif key == ord("v"):
        mode = "VIDEO"
        jerk_hist.clear()
        var_hist.clear()
    elif key == ord("b"):
        bandwidth = max(0.01, bandwidth - 0.05)
    elif key == ord("n"):
        bandwidth = min(1.0, bandwidth + 0.05)
    elif key == ord("q"):
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if mode == "VIDEO":
        noise = np.random.normal(0, (1 - bandwidth) * 0.02, rgb.shape)
        rgb = np.clip(rgb + noise * 255, 0, 255).astype(np.uint8)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    phase = "NO DATA"
    prediction = 0.0

    res = landmarker.detect(mp_image)
    if res.face_landmarks:
        nose = res.face_landmarks[0][1]
        j = jerk(nose.y)
        jerk_hist.append(j)

        if len(jerk_hist) > 6:
            v = np.var(jerk_hist)
            var_hist.append(v)

            prediction = min(1.0, np.mean(var_hist) * 60000)

            if prediction < 0.3:
                phase = "STABLE"
            elif prediction < 0.6:
                phase = "WARNING"
            elif prediction < 0.85:
                phase = "IMMINENT"
            else:
                phase = "COLLAPSE"

        h, w, _ = frame.shape
        cv2.circle(frame, (int(nose.x*w), int(nose.y*h)), 5, (0,255,0), -1)

    # ===============================
    # UI
    # ===============================
    color = (0,255,0)
    if phase == "WARNING": color = (0,255,255)
    if phase == "IMMINENT": color = (0,165,255)
    if phase == "COLLAPSE": color = (0,0,255)

    cv2.putText(frame, f"MODE: {mode}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    cv2.putText(frame, f"BW: {bandwidth:.2f}", (20,80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)
    cv2.putText(frame, f"PHASE: {phase}", (20,120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Prediction bar
    bar = int(prediction * 300)
    cv2.rectangle(frame, (20,150), (20+bar,165), color, -1)
    cv2.putText(frame, f"P(collapse): {prediction:.2f}", (20,190),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("IRIS — Predictive Collapse", frame)

cap.release()
cv2.destroyAllWindows()