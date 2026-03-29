import cv2
import time
import numpy as np
from collections import deque

# ===============================
# CONFIG
# ===============================
WINDOW = 15
STABLE_JERK = 0.02
JITTER_JERK = 0.08

bandwidth_kbps = 0.12
MODE = "IRIS"

# ===============================
# STATE
# ===============================
errors = deque(maxlen=WINDOW)
jerks = deque(maxlen=WINDOW)
last_error = None
last_last_error = None

# ===============================
# PHASE LOGIC
# ===============================
def compute_phase(avg_jerk):
    if avg_jerk < STABLE_JERK:
        return "STABLE"
    elif avg_jerk < JITTER_JERK:
        return "JITTER"
    else:
        return "COLLAPSE"

# ===============================
# CAMERA
# ===============================
cap = cv2.VideoCapture(0)
print("\nIRIS — PHASE CURVE LIVE")
print("b/B → bandwidth down/up")
print("i → IRIS | v → VIDEO")
print("q → quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ====== SIMULATED ERROR ======
    base_error = np.std(gray) / 255.0

    if MODE == "VIDEO":
        noise = np.random.normal(0, 0.08 / max(bandwidth_kbps, 0.01))
        error = base_error + abs(noise)
    else:
        noise = np.random.normal(0, 0.015 / max(bandwidth_kbps, 0.01))
        error = base_error * 0.3 + abs(noise)

    errors.append(error)

    # ====== JERK ======
    if last_last_error is not None:
        jerk = abs(error - 2 * last_error + last_last_error)
        jerks.append(jerk)

    last_last_error = last_error
    last_error = error

    avg_jerk = np.mean(jerks) if jerks else 0.0
    phase = compute_phase(avg_jerk)

    # ===============================
    # DISPLAY
    # ===============================
    color = (0, 255, 0)
    if phase == "JITTER":
        color = (0, 165, 255)
    elif phase == "COLLAPSE":
        color = (0, 0, 255)

    cv2.putText(frame, f"MODE: {MODE}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"Bandwidth: {bandwidth_kbps:.2f} kbps", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"Phase: {phase}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    cv2.putText(frame, f"Avg Jerk: {avg_jerk:.4f}", (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

    cv2.imshow("IRIS Phase Curve", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("b"):
        bandwidth_kbps = max(0.01, bandwidth_kbps - 0.01)
    elif key == ord("B"):
        bandwidth_kbps += 0.01
    elif key == ord("i"):
        MODE = "IRIS"
    elif key == ord("v"):
        MODE = "VIDEO"

cap.release()
cv2.destroyAllWindows()