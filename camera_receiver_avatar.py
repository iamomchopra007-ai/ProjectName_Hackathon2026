import cv2
import json
import numpy as np
import time
import os

print("IRIS RECEIVER — INTENT ONLY (BANDWIDTH PROOF)")
print("Press Q to quit")

W, H = 600, 600
cx, cy = W // 2, H // 2
alpha = 0.2

bytes_total = 0
updates = 0
last_size = 0
start = time.time()
last_seen = time.time()

while True:
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    try:
        stat = os.stat("intent.json")
        delta = stat.st_size - last_size
        if delta > 0:
            bytes_total += delta
            last_size = stat.st_size

        with open("intent.json", "r") as f:
            data = json.load(f)

        tx = int(data["x"] * W)
        ty = int(data["y"] * H)

        updates += 1
        last_seen = time.time()

    except:
        tx, ty = cx, cy

    if time.time() - last_seen < 0.4:
        cx = int(cx * (1 - alpha) + tx * alpha)
        cy = int(cy * (1 - alpha) + ty * alpha)

    # face outline
    cv2.circle(canvas, (W//2, H//2), 220, (200, 200, 200), 2)

    # eyes
    cv2.circle(canvas, (cx - 30, cy - 20), 10, (255, 255, 255), -1)
    cv2.circle(canvas, (cx + 30, cy - 20), 10, (255, 255, 255), -1)

    # nose
    cv2.circle(canvas, (cx, cy), 4, (0, 0, 255), -1)

    elapsed = max(time.time() - start, 0.001)
    kbps = (bytes_total * 8) / 1000 / elapsed
    ups = updates / elapsed

    cv2.putText(canvas, "IRIS RECEIVER (INTENT ONLY)", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(canvas, f"Bandwidth: {kbps:.4f} kbps", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(canvas, f"Updates/sec: {ups:.1f}", (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(canvas, "No frames | No pixels | No stream",
                (20, H - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("IRIS Receiver Avatar", canvas)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()