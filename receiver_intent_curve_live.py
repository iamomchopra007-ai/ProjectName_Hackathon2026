import socket
import json
import time
import cv2
import numpy as np
from collections import deque

# ===============================
# NETWORK
# ===============================
UDP_IP = "0.0.0.0"
UDP_PORT = 5056   # separate port from presence proxy

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.setblocking(False)

# ===============================
# DATA BUFFERS
# ===============================
MAX_POINTS = 200
bandwidths = deque(maxlen=MAX_POINTS)
errors = deque(maxlen=MAX_POINTS)

current_bw = 0.05
current_err = 0.0
phase = 0

# ===============================
# WINDOW
# ===============================
W, H = 700, 500
cv2.namedWindow("IRIS — Intent Saturation Curve (LIVE)")

# ===============================
# LOOP
# ===============================
while True:
    try:
        data, _ = sock.recvfrom(2048)
        intent = json.loads(data.decode())

        jerk = abs(intent.get("jerk", 0.0))
        current_bw = intent.get("bandwidth_kbps", current_bw)
        phase = intent.get("phase", 0)

        instability = 1.0 if phase == 0 else 1.8 if phase == 1 else 3.0
        current_err = jerk * instability

        bandwidths.append(current_bw)
        errors.append(current_err)

    except BlockingIOError:
        pass

    # -------- DRAW --------
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    # Axes
    cv2.line(canvas, (60, 440), (660, 440), (200, 200, 200), 1)
    cv2.line(canvas, (60, 60), (60, 440), (200, 200, 200), 1)

    # Labels
    cv2.putText(canvas, "Bandwidth (kbps)", (260, 480),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.putText(canvas, "Error", (10, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    # Plot points
    if bandwidths:
        max_bw = max(max(bandwidths), 0.12)
        max_err = max(max(errors), 0.01)

        pts = []
        for bw, err in zip(bandwidths, errors):
            x = int(60 + (bw / max_bw) * 580)
            y = int(440 - (err / max_err) * 380)
            pts.append((x, y))

        for p in pts:
            cv2.circle(canvas, p, 2, (0, 200, 255), -1)

        # Current point (highlight)
        cx = int(60 + (current_bw / max_bw) * 580)
        cy = int(440 - (current_err / max_err) * 380)
        cv2.circle(canvas, (cx, cy), 6, (0, 0, 255), -1)

        # Knee annotation
        cv2.putText(canvas, "Intent Saturation",
                    (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

    # Status
    cv2.putText(canvas,
                f"BW: {current_bw:.3f} kbps | ERR: {current_err:.3f}",
                (80, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("IRIS — Intent Saturation Curve (LIVE)", canvas)

    if cv2.waitKey(16) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()