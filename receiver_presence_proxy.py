import socket
import json
import math
import time
import cv2
import numpy as np

# ===============================
# RECEIVER CONFIG
# ===============================
UDP_IP = "0.0.0.0"
UDP_PORT = 5055

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.setblocking(False)

# ===============================
# STATE
# ===============================
state = {
    "yaw": 0.0,
    "pitch": 0.0,
    "roll": 0.0,
    "blink": 0.0,
    "velocity": 0.0,
    "jerk": 0.0,
    "phase": 0,
    "last_time": time.time()
}

# ===============================
# WINDOW
# ===============================
W, H = 600, 600
cv2.namedWindow("IRIS — Receiver Presence (NO CAMERA)")

# ===============================
# MAIN LOOP
# ===============================
while True:
    # -------- RECEIVE INTENT --------
    try:
        data, _ = sock.recvfrom(2048)
        intent = json.loads(data.decode())

        for k in state:
            if k in intent:
                state[k] = intent[k]

        state["last_time"] = time.time()

    except BlockingIOError:
        pass

    # -------- RENDER --------
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    cx, cy = W // 2, H // 2

    # Head position from yaw/pitch
    hx = int(cx + state["yaw"] * 120)
    hy = int(cy + state["pitch"] * 120)

    # Phase color
    if state["phase"] == 0:
        color = (0, 255, 0)
        phase_name = "STABLE"
    elif state["phase"] == 1:
        color = (0, 255, 255)
        phase_name = "UNSTABLE"
    else:
        color = (0, 0, 255)
        phase_name = "COLLAPSE"

    # Head
    cv2.circle(canvas, (hx, hy), 80, color, 2)

    # Eyes
    blink = max(0.0, min(1.0, state["blink"]))
    eye_open = int(8 * (1.0 - blink))

    cv2.line(canvas, (hx - 30, hy - 10), (hx - 30, hy - 10 + eye_open), color, 2)
    cv2.line(canvas, (hx + 30, hy - 10), (hx + 30, hy - 10 + eye_open), color, 2)

    # Gaze dot
    gaze_x = int(hx + state["yaw"] * 40)
    gaze_y = int(hy + state["pitch"] * 40)
    cv2.circle(canvas, (gaze_x, gaze_y), 4, color, -1)

    # Text
    cv2.putText(canvas, "MODE: RECEIVER (NO CAMERA)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.putText(canvas, f"PHASE: {phase_name}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.putText(canvas, f"JERK: {state['jerk']:.3f}", (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("IRIS — Receiver Presence (NO CAMERA)", canvas)

    if cv2.waitKey(16) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()