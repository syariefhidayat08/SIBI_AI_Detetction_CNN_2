import cv2
import imutils
import numpy as np
import os
import string
from datetime import datetime

# =========================================================
# CONFIG
# =========================================================

# Semua class:
# 0-9 dan A-Z
CLASSES = [str(i) for i in range(10)] + list(string.ascii_uppercase)

MODE = "Training"  # Training / Validation

TRAINING_LIMIT = 1000
VALIDATION_LIMIT = 200

# Ganti class di sini
CURRENT_CLASS = "S"

SAVE_LIMIT = TRAINING_LIMIT if MODE == "Training" else VALIDATION_LIMIT

# ROI AREA
TOP = 80
RIGHT = 420
BOTTOM = 430
LEFT = 700

# Background
bg = None

# =========================================================
# CREATE DATASET FOLDER
# =========================================================

SAVE_DIR = f"Dataset/{MODE}/{CURRENT_CLASS}"

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================================================
# BACKGROUND AVERAGE
# =========================================================

def run_avg(image, aWeight):
    global bg

    if bg is None:
        bg = image.copy().astype("float")
        return

    cv2.accumulateWeighted(image, bg, aWeight)

# =========================================================
# HAND SEGMENTATION
# =========================================================

def segment(image, threshold=25):
    global bg

    diff = cv2.absdiff(bg.astype("uint8"), image)

    thresholded = cv2.threshold(
        diff,
        threshold,
        255,
        cv2.THRESH_BINARY
    )[1]

    thresholded = cv2.erode(thresholded, None, iterations=2)
    thresholded = cv2.dilate(thresholded, None, iterations=2)

    cnts, _ = cv2.findContours(
        thresholded.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(cnts) == 0:
        return None

    segmented = max(cnts, key=cv2.contourArea)

    return (thresholded, segmented)

# =========================================================
# DRAW UI PANEL
# =========================================================

def draw_ui(frame, image_num, recording, calibrated):
    h, w = frame.shape[:2]

    # HEADER
    cv2.rectangle(frame, (0, 0), (w, 60), (30, 30, 30), -1)

    title = "HAND DATASET COLLECTOR"
    cv2.putText(
        frame,
        title,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    # INFO PANEL
    cv2.rectangle(frame, (10, 70), (350, 250), (40, 40, 40), -1)

    status = "ON" if recording else "OFF"
    bg_status = "READY" if calibrated else "CALIBRATING"

    info = [
        f"Mode      : {MODE}",
        f"Class     : {CURRENT_CLASS}",
        f"Saved     : {image_num}/{SAVE_LIMIT}",
        f"Recording : {status}",
        f"Background: {bg_status}",
    ]

    y = 105

    for text in info:
        cv2.putText(
            frame,
            text,
            (25, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        y += 30

    # HOTKEYS
    cv2.putText(
        frame,
        "[S] Start Recording",
        (25, 225),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        1
    )

    cv2.putText(
        frame,
        "[P] Pause Recording",
        (190, 225),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1
    )

    cv2.putText(
        frame,
        "[R] Reset Background",
        (25, 245),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 0),
        1
    )

    cv2.putText(
        frame,
        "[Q] Quit",
        (220, 245),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        1
    )

# =========================================================
# MAIN
# =========================================================

def main():

    global bg

    aWeight = 0.5

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("[ERROR] Camera tidak ditemukan!")
        return

    num_frames = 0
    image_num = len(os.listdir(SAVE_DIR))

    start_recording = False

    print("=" * 50)
    print("AVAILABLE CLASSES")
    print(CLASSES)
    print("=" * 50)

    while True:

        grabbed, frame = camera.read()

        if not grabbed:
            print("[WARNING] Gagal membaca camera.")
            break

        # Resize
        frame = imutils.resize(frame, width=900)

        # Mirror
        frame = cv2.flip(frame, 1)

        clone = frame.copy()

        # ROI
        roi = frame[TOP:BOTTOM, RIGHT:LEFT]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # =====================================================
        # BACKGROUND CALIBRATION
        # =====================================================

        if num_frames < 30:

            run_avg(gray, aWeight)

            cv2.putText(
                clone,
                f"Calibrating Background... {num_frames}/30",
                (430, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

        else:

            hand = segment(gray)

            if hand is not None:

                thresholded, segmented = hand

                # Draw contour
                cv2.drawContours(
                    clone,
                    [segmented + (RIGHT, TOP)],
                    -1,
                    (0, 0, 255),
                    2
                )

                # SAVE IMAGE
                if start_recording and image_num < SAVE_LIMIT:

                    filename = f"{CURRENT_CLASS}_{image_num}.png"

                    save_path = os.path.join(SAVE_DIR, filename)

                    cv2.imwrite(save_path, thresholded)

                    image_num += 1

                    # Flash effect
                    cv2.rectangle(
                        clone,
                        (0, 0),
                        (clone.shape[1], clone.shape[0]),
                        (0, 255, 0),
                        10
                    )

                # Show threshold
                cv2.imshow("Thresholded", thresholded)

        # ROI BOX
        color = (0, 255, 0) if start_recording else (255, 255, 0)

        cv2.rectangle(
            clone,
            (RIGHT, TOP),
            (LEFT, BOTTOM),
            color,
            3
        )

        # Draw UI
        draw_ui(
            clone,
            image_num,
            start_recording,
            num_frames >= 30
        )

        # AUTO STOP
        if image_num >= SAVE_LIMIT:
            start_recording = False

            cv2.putText(
                clone,
                "DATASET LIMIT REACHED!",
                (420, 470),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

        # SHOW CAMERA
        cv2.imshow("Hand Dataset Collector", clone)

        keypress = cv2.waitKey(1) & 0xFF

        # =====================================================
        # HOTKEYS
        # =====================================================

        # START
        if keypress == ord("s"):
            start_recording = True

        # PAUSE
        elif keypress == ord("p"):
            start_recording = False

        # RESET BG
        elif keypress == ord("r"):
            bg = None
            num_frames = 0
            print("[INFO] Background reset!")

        # QUIT
        elif keypress == ord("q"):
            break

        num_frames += 1

    # =========================================================
    # RELEASE
    # =========================================================

    camera.release()
    cv2.destroyAllWindows()

# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()