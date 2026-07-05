# OpenCV Vision Applications
> 5 real-time computer vision projects | Python + OpenCV | 20+ fps

---

## Quick Setup (VS Code)

```bash
# 1. Create virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Applications

| # | App | File | Algorithm | FPS |
|---|-----|------|-----------|-----|
| 1 | Object Detection | `app1_object_detection/object_detection.py` | YOLOv3 + NMS | 22 |
| 2 | Face Detection   | `app2_face_detection/face_detection.py`     | Haar Cascades | 28 |
| 3 | Motion Tracking  | `app3_motion_tracking/motion_tracking.py`   | MOG2 + Kalman | 30 |
| 4 | Image Classification | `app4_image_classification/image_classification.py` | HOG + SVM | 20 |
| 5 | Lane Detection   | `app5_lane_detection/lane_detection.py`     | Canny + Hough | 25 |

---

## Run Each App

```bash
# App 1 - Object Detection (downloads YOLOv3 weights ~236MB on first run)
python app1_object_detection/object_detection.py

# App 2 - Face Detection (no downloads, uses built-in OpenCV cascades)
python app2_face_detection/face_detection.py

# App 3 - Motion Tracking
python app3_motion_tracking/motion_tracking.py

# App 4 - HOG + SVM Classification
python app4_image_classification/image_classification.py

# App 5 - Lane Detection
python app5_lane_detection/lane_detection.py
```

---

## Key Controls (all apps)

| Key | Action |
|-----|--------|
| Q | Quit |
| S | Screenshot |
| M | Toggle algorithm (App 3) |
| E | Toggle edge view (App 5) |
| 1-5 | Select class (App 4) |
| C | Capture sample (App 4) |
| T | Train model (App 4) |

---

## Notes for App 1 (YOLOv3)
- First run downloads `yolov3.weights` (~236 MB) automatically
- If download is slow, you can manually place `yolov3.weights` in the `app1_object_detection/` folder
- Download from: https://pjreddie.com/media/files/yolov3.weights

## Notes for App 4 (HOG+SVM)
- Runs in HOG visualization demo mode without training data
- To train: Press 1-5 to select class → C to capture 30+ samples → T to train
- Recognizes 5 hand gestures: fist, open_hand, peace, thumbs_up, pointing
