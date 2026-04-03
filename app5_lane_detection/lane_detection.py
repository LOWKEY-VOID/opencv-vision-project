"""
App 5: Lane & Edge Detection using Canny + Hough Transform
============================================================
Detects lane lines from webcam or video file using:
  - Gaussian Blur → Canny Edge Detection → ROI Mask → Hough Lines

Can be used on a road video or any straight-line scene.
Works on webcam (tilt to show floor lines / table edges / books).

Requirements:
    pip install opencv-python numpy

Controls:
    Q          - Quit
    E          - Toggle edge mask overlay
    V          - Load video file (enter path in terminal)
    1 / 2      - Decrease / Increase Canny threshold
    3 / 4      - Decrease / Increase Hough threshold
    S          - Screenshot
    R          - Reset parameters
"""

import cv2
import numpy as np
import time
import os

# ── Tunable parameters ────────────────────────────────────────────────────────
params = {
    "canny_low":    50,
    "canny_high":   150,
    "hough_thresh": 50,
    "hough_min_len": 40,
    "hough_max_gap": 25,
    "blur_ksize":   5,
}

# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(frame, ksize):
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    return blurred

def canny_edges(blurred, low, high):
    return cv2.Canny(blurred, low, high)

def roi_mask(edges, vertices):
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(edges, mask)

def hough_lines(masked, rho=1, theta=np.pi/180,
                threshold=50, min_len=40, max_gap=25):
    return cv2.HoughLinesP(
        masked, rho, theta, threshold,
        minLineLength=min_len, maxLineGap=max_gap
    )

# ── Line averaging (left / right lane) ───────────────────────────────────────
def average_lines(frame, lines):
    left_pts, right_pts = [], []
    h, w = frame.shape[:2]

    if lines is None:
        return [], []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        if abs(slope) < 0.3:   # Ignore near-horizontal lines
            continue
        if slope < 0:
            left_pts.append((slope, intercept))
        else:
            right_pts.append((slope, intercept))

    def make_coords(slope, intercept):
        y1 = h
        y2 = int(h * 0.55)
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return [(x1, y1, x2, y2)]

    result = []
    if left_pts:
        avg = np.average(left_pts, axis=0)
        result += make_coords(*avg)
    if right_pts:
        avg = np.average(right_pts, axis=0)
        result += make_coords(*avg)

    return result

# ── Draw lanes ────────────────────────────────────────────────────────────────
def draw_lanes(frame, avg_lines, raw_lines=None):
    lane_img = np.zeros_like(frame)
    h, w = frame.shape[:2]

    # Draw averaged lane lines
    for (x1, y1, x2, y2) in avg_lines:
        cv2.line(lane_img, (x1, y1), (x2, y2), (0, 200, 100), 8)

    # Fill lane polygon
    if len(avg_lines) == 2:
        (lx1, ly1, lx2, ly2) = avg_lines[0]
        (rx1, ry1, rx2, ry2) = avg_lines[1]
        pts = np.array([[lx1, ly1], [lx2, ly2],
                         [rx2, ry2], [rx1, ry1]], dtype=np.int32)
        cv2.fillPoly(lane_img, [pts], (0, 100, 50))

    # Draw raw Hough lines (thin, faded)
    if raw_lines is not None:
        for line in raw_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(lane_img, (x1, y1), (x2, y2), (0, 80, 200), 1)

    return cv2.addWeighted(frame, 1.0, lane_img, 0.5, 0)

# ── Parameter trackbar window ─────────────────────────────────────────────────
def create_trackbars():
    cv2.namedWindow("Parameters")
    cv2.createTrackbar("Canny Low",    "Parameters", params["canny_low"],    300, lambda v: params.update({"canny_low": max(1,v)}))
    cv2.createTrackbar("Canny High",   "Parameters", params["canny_high"],   500, lambda v: params.update({"canny_high": max(1,v)}))
    cv2.createTrackbar("Hough Thresh", "Parameters", params["hough_thresh"], 200, lambda v: params.update({"hough_thresh": max(1,v)}))
    cv2.createTrackbar("Min Length",   "Parameters", params["hough_min_len"],200, lambda v: params.update({"hough_min_len": max(1,v)}))
    cv2.createTrackbar("Max Gap",      "Parameters", params["hough_max_gap"],100, lambda v: params.update({"hough_max_gap": max(1,v)}))

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    show_edges = False
    create_trackbars()
    os.makedirs("screenshots", exist_ok=True)

    fps_counter, fps_start, fps = 0, time.time(), 0
    print("[INFO] Running. Q=Quit | E=Edge overlay | S=Screenshot")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        # ── Pipeline ──────────────────────────────────────────────────────
        blurred = preprocess(frame, params["blur_ksize"])
        edges   = canny_edges(blurred, params["canny_low"], params["canny_high"])

        # Trapezoidal ROI — lower portion of frame (road area)
        roi_verts = np.array([[
            (0,         h),
            (w * 0.45,  h * 0.55),
            (w * 0.55,  h * 0.55),
            (w,         h)
        ]], dtype=np.int32)
        masked_edges = roi_mask(edges, roi_verts)

        lines     = hough_lines(masked_edges,
                                threshold=params["hough_thresh"],
                                min_len=params["hough_min_len"],
                                max_gap=params["hough_max_gap"])
        avg       = average_lines(frame, lines)
        result    = draw_lanes(frame, avg, lines)

        # Draw ROI boundary
        cv2.polylines(result, roi_verts, True, (100, 100, 100), 1)

        # Edge overlay (bottom-right inset)
        if show_edges:
            edge_color = cv2.cvtColor(masked_edges, cv2.COLOR_GRAY2BGR)
            small = cv2.resize(edge_color, (w // 4, h // 4))
            result[h - h//4:, w - w//4:] = small
            cv2.putText(result, "Canny edges", (w - w//4, h - h//4 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        # ── FPS & HUD ─────────────────────────────────────────────────────
        fps_counter += 1
        if fps_counter % 30 == 0:
            fps = 30 / (time.time() - fps_start)
            fps_start = time.time()

        n_lines = len(lines) if lines is not None else 0
        cv2.rectangle(result, (0, 0), (400, 80), (0, 0, 0), -1)
        cv2.putText(result, f"FPS:{fps:.0f}  Hough lines:{n_lines}  Avg lanes:{len(avg)}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(result,
                    f"Canny:{params['canny_low']}/{params['canny_high']}  "
                    f"Hough:{params['hough_thresh']}  "
                    f"MinLen:{params['hough_min_len']}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.putText(result, "E=edges  Q=quit  S=screenshot",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        cv2.imshow("Lane Detection | Canny + Hough | Q=Quit", result)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('e'):
            show_edges = not show_edges
        elif key == ord('s'):
            fname = f"screenshots/lane_{int(time.time())}.jpg"
            cv2.imwrite(fname, result)
            print(f"[INFO] Screenshot: {fname}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
