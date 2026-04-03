"""
App 2: Face Detection using Haar Cascades
==========================================
Detects faces, eyes, and smile in real-time from webcam.
Uses OpenCV's built-in Haar Cascade classifiers (no extra downloads).

Requirements:
    pip install opencv-python numpy

Press 'Q' to quit | Press 'S' to save a screenshot.
"""

import cv2
import numpy as np
import os
import time

# ── Load Haar Cascades (bundled with OpenCV) ──────────────────────────────────
def load_cascades():
    cascade_dir = cv2.data.haarcascades

    face_cascade  = cv2.CascadeClassifier(cascade_dir + "haarcascade_frontalface_default.xml")
    eye_cascade   = cv2.CascadeClassifier(cascade_dir + "haarcascade_eye.xml")
    smile_cascade = cv2.CascadeClassifier(cascade_dir + "haarcascade_smile.xml")
    profile_cascade = cv2.CascadeClassifier(cascade_dir + "haarcascade_profileface.xml")

    return face_cascade, eye_cascade, smile_cascade, profile_cascade


# ── Detection ─────────────────────────────────────────────────────────────────
def detect_faces(frame, face_cascade, eye_cascade, smile_cascade, profile_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Histogram equalization improves detection in low light
    gray = cv2.equalizeHist(gray)

    # Detect frontal faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Detect profile faces (side view)
    profiles = profile_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(60, 60)
    )

    result = frame.copy()
    face_count = 0

    # Draw frontal faces
    for (x, y, w, h) in faces:
        face_count += 1
        cv2.rectangle(result, (x, y), (x+w, y+h), (255, 165, 0), 2)
        cv2.putText(result, f"Face #{face_count}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

        # ROI for eye detection (upper half of face only)
        roi_gray  = gray[y:y+h//2, x:x+w]
        roi_color = result[y:y+h//2, x:x+w]

        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=10,
            minSize=(20, 20)
        )
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 1)
            cv2.circle(roi_color, (ex + ew//2, ey + eh//2), 3, (0, 255, 0), -1)

        # Smile detection (lower half of face)
        roi_gray_lower  = gray[y+h//2:y+h, x:x+w]
        roi_color_lower = result[y+h//2:y+h, x:x+w]
        smiles = smile_cascade.detectMultiScale(
            roi_gray_lower,
            scaleFactor=1.7,
            minNeighbors=22,
            minSize=(25, 15)
        )
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color_lower, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 1)
            cv2.putText(result, "Smile :)", (x, y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Draw profile faces
    for (x, y, w, h) in profiles:
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.putText(result, "Profile", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    return result, len(faces) + len(profiles)


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    print("[INFO] Loading Haar Cascade classifiers ...")
    face_cascade, eye_cascade, smile_cascade, profile_cascade = load_cascades()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("[INFO] Running. Press 'Q' to quit | 'S' to screenshot.")
    fps_counter, fps_start = 0, time.time()
    fps = 0

    os.makedirs("screenshots", exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result, face_count = detect_faces(frame, face_cascade, eye_cascade,
                                           smile_cascade, profile_cascade)

        # Update FPS every 30 frames
        fps_counter += 1
        if fps_counter % 30 == 0:
            fps = 30 / (time.time() - fps_start)
            fps_start = time.time()

        # HUD overlay
        cv2.rectangle(result, (0, 0), (280, 75), (0, 0, 0), -1)
        cv2.putText(result, f"FPS       : {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(result, f"Faces     : {face_count}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(result, "Method: Haar + ROI", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        cv2.imshow("Haar Cascade Face Detection | Q=Quit  S=Screenshot", result)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f"screenshots/face_{int(time.time())}.jpg"
            cv2.imwrite(fname, result)
            print(f"[INFO] Screenshot saved: {fname}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
