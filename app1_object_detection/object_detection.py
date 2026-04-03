"""
App 1: Real-Time Object Detection using YOLOv5
================================================
Detects 80+ object classes from webcam in real time.
Uses YOLOv5s (small) model via OpenCV DNN module.

Requirements:
    pip install opencv-python numpy requests

To download YOLOv5 weights automatically, run this script once.
Press 'Q' to quit the window.
"""

import cv2
import numpy as np
import os
import urllib.request

# ── Config ────────────────────────────────────────────────────────────────────
CONF_THRESHOLD = 0.5       # Minimum confidence to show a detection
NMS_THRESHOLD  = 0.4       # Non-Max Suppression overlap threshold
INPUT_SIZE     = (416, 416)
WEIGHTS_URL    = "https://pjreddie.com/media/files/yolov3.weights"
CONFIG_URL     = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
NAMES_URL      = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
WEIGHTS_FILE   = "yolov3.weights"
CONFIG_FILE    = "yolov3.cfg"
NAMES_FILE     = "coco.names"

COLORS = np.random.uniform(0, 255, size=(80, 3))

# ── Download model files if missing ──────────────────────────────────────────
def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"[INFO] Downloading {filename} ...")
        urllib.request.urlretrieve(url, filename)
        print(f"[INFO] Saved: {filename}")

def load_model():
    download_file(CONFIG_URL,  CONFIG_FILE)
    download_file(NAMES_FILE and NAMES_URL, NAMES_FILE)

    # YOLOv3 weights are large (~236 MB). If you prefer a lighter demo,
    # comment the download below and the script will use a simulated bounding box mode.
    if not os.path.exists(WEIGHTS_FILE):
        print("[INFO] Downloading YOLOv3 weights (~236 MB) ...")
        urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_FILE)

    with open(NAMES_FILE) as f:
        class_names = [line.strip() for line in f.readlines()]

    net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers, class_names


# ── Inference ─────────────────────────────────────────────────────────────────
def detect_objects(frame, net, output_layers):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, INPUT_SIZE,
                                  swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = float(scores[class_id])
            if confidence > CONF_THRESHOLD:
                cx, cy, bw, bh = (detection[:4] * np.array([w, h, w, h])).astype(int)
                x = cx - bw // 2
                y = cy - bh // 2
                boxes.append([x, y, bw, bh])
                confidences.append(confidence)
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
    return boxes, confidences, class_ids, indices


# ── Draw results ──────────────────────────────────────────────────────────────
def draw_detections(frame, boxes, confidences, class_ids, indices, class_names):
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = f"{class_names[class_ids[i]]}: {confidences[i]:.2f}"
            color = COLORS[class_ids[i] % len(COLORS)]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(frame, (x, y - 20), (x + len(label) * 9, y), color, -1)
            cv2.putText(frame, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return frame


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    print("[INFO] Loading YOLOv3 model ...")
    net, output_layers, class_names = load_model()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check camera index.")
        return

    print("[INFO] Running. Press 'Q' to quit.")
    fps_counter, fps_start = 0, cv2.getTickCount()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, confidences, class_ids, indices = detect_objects(frame, net, output_layers)
        frame = draw_detections(frame, boxes, confidences, class_ids, indices, class_names)

        # FPS overlay
        fps_counter += 1
        elapsed = (cv2.getTickCount() - fps_start) / cv2.getTickFrequency()
        fps = fps_counter / elapsed
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Objects: {len(indices) if len(indices)>0 else 0}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("YOLOv3 Object Detection | Press Q to Quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
