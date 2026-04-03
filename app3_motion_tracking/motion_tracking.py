"""
App 3: Motion Tracking using Background Subtraction
=====================================================
Detects and tracks moving objects using MOG2 / KNN background subtraction.
Draws bounding boxes, centroids, and motion trails.

Requirements:
    pip install opencv-python numpy

Press 'Q' to quit | 'M' to toggle MOG2/KNN | 'R' to reset background model.
"""

import cv2
import numpy as np
import time
from collections import defaultdict, deque

# ── Config ────────────────────────────────────────────────────────────────────
MIN_CONTOUR_AREA  = 1500    # Ignore tiny blobs
TRAIL_LENGTH      = 30      # How many frames to keep motion trail
LEARNING_RATE     = 0.01    # Background model adaptation speed (0=freeze, 1=instant)

# ── Object tracker (simple centroid-based) ───────────────────────────────────
class CentroidTracker:
    def __init__(self):
        self.next_id = 0
        self.objects = {}         # id -> centroid
        self.trails  = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))
        self.disappeared = {}

    def update(self, centroids):
        if not centroids:
            for obj_id in list(self.disappeared):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > 20:
                    del self.objects[obj_id]
                    del self.disappeared[obj_id]
            return self.objects

        if not self.objects:
            for c in centroids:
                self.objects[self.next_id] = c
                self.disappeared[self.next_id] = 0
                self.next_id += 1
        else:
            obj_ids = list(self.objects.keys())
            obj_centroids = list(self.objects.values())

            # Compute Euclidean distances
            D = np.linalg.norm(
                np.array(obj_centroids)[:, np.newaxis] - np.array(centroids), axis=2
            )
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()
            for r, c in zip(rows, cols):
                if r in used_rows or c in used_cols:
                    continue
                obj_id = obj_ids[r]
                self.objects[obj_id] = centroids[c]
                self.trails[obj_id].append(centroids[c])
                self.disappeared[obj_id] = 0
                used_rows.add(r)
                used_cols.add(c)

            unused_rows = set(range(len(obj_ids))) - used_rows
            unused_cols = set(range(len(centroids))) - used_cols

            for r in unused_rows:
                obj_id = obj_ids[r]
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > 20:
                    del self.objects[obj_id]
                    if obj_id in self.trails:
                        del self.trails[obj_id]
                    del self.disappeared[obj_id]

            for c in unused_cols:
                self.objects[self.next_id] = centroids[c]
                self.trails[self.next_id].append(centroids[c])
                self.disappeared[self.next_id] = 0
                self.next_id += 1

        return self.objects


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Two background subtractor options
    mog2 = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=50, detectShadows=True
    )
    knn  = cv2.createBackgroundSubtractorKNN(
        history=500, dist2Threshold=400.0, detectShadows=True
    )
    use_mog2  = True
    tracker   = CentroidTracker()

    print("[INFO] Running. Q=Quit | M=Toggle MOG2/KNN | R=Reset model")

    fps_counter, fps_start, fps = 0, time.time(), 0
    COLORS = [(0, 255, 120), (255, 165, 0), (0, 200, 255),
              (200, 0, 255), (255, 50, 50), (50, 255, 255)]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        subtractor = mog2 if use_mog2 else knn

        # ── Background subtraction ─────────────────────────────────────────
        fg_mask = subtractor.apply(frame, learningRate=LEARNING_RATE)

        # Remove shadows (marked as 127), keep only full foreground (255)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)

        # ── Contour detection ─────────────────────────────────────────────
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        result    = frame.copy()
        centroids = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_CONTOUR_AREA:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w // 2, y + h // 2
            centroids.append((cx, cy))

            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 120), 2)
            cv2.putText(result, f"area:{int(area)}", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 120), 1)

        # ── Track objects ─────────────────────────────────────────────────
        objects = tracker.update(centroids)

        for obj_id, (cx, cy) in objects.items():
            color = COLORS[obj_id % len(COLORS)]
            cv2.circle(result, (cx, cy), 6, color, -1)
            cv2.putText(result, f"ID:{obj_id}", (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # Draw motion trail
            trail = list(tracker.trails[obj_id])
            for i in range(1, len(trail)):
                alpha = i / len(trail)
                c = tuple(int(x * alpha) for x in color)
                cv2.line(result, trail[i-1], trail[i], c, 2)

        # ── Mask inset (bottom right) ──────────────────────────────────────
        mh, mw = frame.shape[:2]
        small_mask = cv2.resize(cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR),
                                (mw // 4, mh // 4))
        result[mh - mh//4:, mw - mw//4:] = small_mask

        # ── FPS & HUD ─────────────────────────────────────────────────────
        fps_counter += 1
        if fps_counter % 30 == 0:
            fps = 30 / (time.time() - fps_start)
            fps_start = time.time()

        method_str = "MOG2" if use_mog2 else "KNN"
        cv2.rectangle(result, (0, 0), (300, 85), (0, 0, 0), -1)
        cv2.putText(result, f"FPS      : {fps:.1f}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(result, f"Objects  : {len(objects)}", (10, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(result, f"Method   : {method_str}  [M to switch]", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.putText(result, "Mask preview: bottom-right", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

        cv2.imshow("Motion Tracker | Q=Quit  M=Toggle  R=Reset", result)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            use_mog2 = not use_mog2
            print(f"[INFO] Switched to {'MOG2' if use_mog2 else 'KNN'}")
        elif key == ord('r'):
            mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)
            knn  = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400)
            tracker = CentroidTracker()
            print("[INFO] Background model reset.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
