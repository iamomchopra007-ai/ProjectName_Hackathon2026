import cv2
import json
import time
import numpy as np

print("IRIS SENDER — INTENT ONLY (NO VIDEO SENT)")
print("Press Q to quit")

cap = cv2.VideoCapture(0)

ret, prev = cap.read()
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

# pick a single strong point near center
h, w = prev_gray.shape
p0 = np.array([[[w//2, h//2]]], dtype=np.float32)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    p1, st, err = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, p0, None,
        winSize=(21, 21), maxLevel=3
    )

    if st[0][0] == 1:
        x, y = p1[0][0]
        nx, ny = x / w, y / h

        intent = {
            "x": round(float(nx), 4),
            "y": round(float(ny), 4),
            "t": round(time.time(), 4)
        }

        with open("intent.json", "w") as f:
            json.dump(intent, f)

        cv2.circle(frame, (int(x), int(y)), 6, (0, 0, 255), -1)
        p0 = p1

    prev_gray = gray.copy()

    cv2.putText(
        frame,
        "IRIS SENDER (NO VIDEO SENT)",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("IRIS Sender", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()