"""
App 4: Image Classification using HOG Features + SVM
======================================================
Classifies live webcam frames into 5 gesture classes.
Pipeline: Grayscale → Resize → HOG features → PCA → SVM predict

DEMO MODE (no training data needed):
  - The script runs in DEMO mode showing HOG visualization and feature extraction.
  - To train the full classifier, collect images per class (see TRAINING section below).

Requirements:
    pip install opencv-python numpy scikit-learn joblib

Controls:
    Q         - Quit
    C         - Capture frame for training (see classes below)
    1-5       - Set current capture class
    T         - Train SVM from captured images
    S         - Save trained model
    L         - Load saved model
"""

import cv2
import numpy as np
import os
import time
import pickle
from pathlib import Path

try:
    from sklearn.svm import SVC
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("[WARN] scikit-learn not found. Running HOG visualization mode only.")
    print("       Install with: pip install scikit-learn joblib")

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE      = (64, 128)   # HOG window size
HOG_WIN_SIZE  = (64, 128)
HOG_BLOCK     = (16, 16)
HOG_BLOCK_STR = (8, 8)
HOG_CELL      = (8, 8)
HOG_BINS      = 9
PCA_COMPONENTS = 128
CLASSES       = ["fist", "open_hand", "peace", "thumbs_up", "pointing"]
CLASS_COLORS  = [(0, 165, 255), (0, 255, 120), (255, 50, 200),
                 (50, 200, 255), (255, 200, 50)]
DATA_DIR      = "training_data"
MODEL_FILE    = "hog_svm_model.pkl"

# ── HOG Descriptor ────────────────────────────────────────────────────────────
def create_hog():
    return cv2.HOGDescriptor(
        HOG_WIN_SIZE, HOG_BLOCK, HOG_BLOCK_STR,
        HOG_CELL, HOG_BINS
    )

def extract_features(frame, hog):
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMG_SIZE)
    feats   = hog.compute(resized)
    return feats.flatten()

# ── HOG Visualization (for demo / HR presentation) ───────────────────────────
def visualize_hog(frame):
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 256))

    hog_vis, _ = cv2.HOGDescriptor(
        (128, 256), (16,16), (8,8), (8,8), 9
    ).compute(resized, winStride=(8,8), padding=(0,0))

    cell_w, cell_h = 8, 8
    angle_unit = 360 / 9
    vis = np.zeros((256, 128, 3), dtype=np.uint8)

    for y in range(0, 256, cell_h):
        for x in range(0, 128, cell_w):
            cy, cx = y + cell_h // 2, x + cell_w // 2
            for b in range(9):
                angle = b * angle_unit
                mag   = float(hog_vis[0]) * 3
                rad   = np.radians(angle)
                dx, dy = int(np.cos(rad) * mag), int(np.sin(rad) * mag)
                cv2.line(vis, (cx, cy), (cx + dx, cy + dy), (0, 200, 100), 1)

    return cv2.resize(vis, (frame.shape[1] // 3, frame.shape[0] // 3))

# ── Training ──────────────────────────────────────────────────────────────────
def train_model(data_dir, hog):
    X, y = [], []
    for i, cls in enumerate(CLASSES):
        cls_dir = Path(data_dir) / cls
        if not cls_dir.exists():
            continue
        for img_path in cls_dir.glob("*.jpg"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            feats = extract_features(img, hog)
            X.append(feats)
            y.append(i)

    if len(X) < 10:
        print(f"[WARN] Only {len(X)} samples. Need more data (10+ per class).")
        return None

    X = np.array(X)
    y = np.array(y)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=min(PCA_COMPONENTS, len(X)-1))),
        ("svm",    SVC(kernel="rbf", C=10, gamma="scale", probability=True))
    ])
    model.fit(X, y)
    acc = model.score(X, y)
    print(f"[INFO] Training accuracy: {acc*100:.1f}%  ({len(X)} samples)")
    return model


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    hog = create_hog()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    model = None
    if os.path.exists(MODEL_FILE) and SKLEARN_OK:
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)
        print("[INFO] Loaded saved model.")

    current_class = 0
    sample_counts = {i: 0 for i in range(len(CLASSES))}
    os.makedirs(DATA_DIR, exist_ok=True)
    for cls in CLASSES:
        Path(DATA_DIR, cls).mkdir(exist_ok=True)
        sample_counts[CLASSES.index(cls)] = len(list(Path(DATA_DIR, cls).glob("*.jpg")))

    fps_counter, fps_start, fps = 0, time.time(), 0
    print("[INFO] Running. Q=Quit | 1-5=Select class | C=Capture | T=Train | S=Save | L=Load")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror view
        result = frame.copy()
        h, w = frame.shape[:2]

        # ── ROI: center square ────────────────────────────────────────────
        roi_size = min(h, w) // 2
        roi_x    = w // 2 - roi_size // 2
        roi_y    = h // 2 - roi_size // 2
        roi      = frame[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]

        cv2.rectangle(result, (roi_x, roi_y),
                      (roi_x+roi_size, roi_y+roi_size),
                      CLASS_COLORS[current_class], 2)
        cv2.putText(result, "Place gesture here", (roi_x, roi_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, CLASS_COLORS[current_class], 1)

        # ── HOG visualization (inset, top-right) ─────────────────────────
        hog_vis = visualize_hog(roi)
        hvh, hvw = hog_vis.shape[:2]
        result[10:10+hvh, w-hvw-10:w-10] = hog_vis
        cv2.rectangle(result, (w-hvw-10, 10), (w-10, 10+hvh),
                      (180, 180, 180), 1)
        cv2.putText(result, "HOG features", (w-hvw-10, 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        # ── Classify (if model loaded) ────────────────────────────────────
        pred_label, confidence = "No model", 0.0
        if model is not None and SKLEARN_OK:
            try:
                feats = extract_features(roi, hog).reshape(1, -1)
                pred  = model.predict(feats)[0]
                proba = model.predict_proba(feats)[0]
                pred_label  = CLASSES[pred]
                confidence  = float(proba[pred])

                # Probability bars
                bar_x = 10
                for i, (cls, p) in enumerate(zip(CLASSES, proba)):
                    bar_len = int(p * 150)
                    by = h - 130 + i * 22
                    cv2.rectangle(result, (bar_x, by),
                                  (bar_x + bar_len, by + 16),
                                  CLASS_COLORS[i], -1)
                    cv2.putText(result,
                                f"{cls:12s} {p*100:5.1f}%",
                                (bar_x, by + 13),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                (255, 255, 255), 1)
            except Exception as e:
                pred_label = f"Error: {str(e)[:20]}"

        # ── HUD ───────────────────────────────────────────────────────────
        fps_counter += 1
        if fps_counter % 30 == 0:
            fps = 30 / (time.time() - fps_start)
            fps_start = time.time()

        cv2.rectangle(result, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.putText(result, f"FPS:{fps:.0f}  Pred:{pred_label}  Conf:{confidence*100:.0f}%",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(result, f"Capture class [{current_class+1}]: {CLASSES[current_class]}  "
                            f"Samples: {sample_counts[current_class]}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        cv2.imshow("HOG + SVM Classifier | Q=Quit  1-5=Class  C=Capture  T=Train", result)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
            current_class = key - ord('1')
            print(f"[INFO] Class set to: {CLASSES[current_class]}")
        elif key == ord('c'):
            fname = f"{DATA_DIR}/{CLASSES[current_class]}/{int(time.time()*1000)}.jpg"
            cv2.imwrite(fname, roi)
            sample_counts[current_class] += 1
            print(f"[INFO] Saved sample #{sample_counts[current_class]} for '{CLASSES[current_class]}'")
        elif key == ord('t') and SKLEARN_OK:
            print("[INFO] Training model ...")
            model = train_model(DATA_DIR, hog)
        elif key == ord('s') and model is not None:
            with open(MODEL_FILE, "wb") as f:
                pickle.dump(model, f)
            print(f"[INFO] Model saved: {MODEL_FILE}")
        elif key == ord('l') and SKLEARN_OK:
            if os.path.exists(MODEL_FILE):
                with open(MODEL_FILE, "rb") as f:
                    model = pickle.load(f)
                print("[INFO] Model loaded.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
