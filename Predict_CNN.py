import cv2
import numpy as np
import tensorflow as tf
import string
import imutils

# =========================================================
# CONFIG
# =========================================================

IMG_WIDTH = 100
IMG_HEIGHT = 89

CLASSES = [str(i) for i in range(10)] + list(string.ascii_uppercase)

MODEL_PATH = "TrainedModelCNN/SIBI_AZ09_MODEL.keras"

# ROI
TOP, RIGHT, BOTTOM, LEFT = 80, 420, 430, 700

# Background
bg = None

# =========================================================
# LOAD MODEL
# =========================================================

model = tf.keras.models.load_model(MODEL_PATH)

print("[INFO] Model loaded successfully!")

# =========================================================
# BACKGROUND AVG
# =========================================================

def run_avg(image, aWeight=0.5):
    global bg

    if bg is None:
        bg = image.copy().astype("float")
        return

    cv2.accumulateWeighted(image, bg, aWeight)

# =========================================================
# SEGMENTATION
# =========================================================

def segment(image, threshold=25):
    global bg

    diff = cv2.absdiff(bg.astype("uint8"), image)

    thresholded = cv2.threshold(
        diff, threshold, 255, cv2.THRESH_BINARY
    )[1]

    cnts, _ = cv2.findContours(
        thresholded.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(cnts) == 0:
        return None

    segmented = max(cnts, key=cv2.contourArea)

    return thresholded, segmented

# =========================================================
# PREDICT
# =========================================================

def predict(image):

    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image / 255.0
    image = image.reshape(1, IMG_HEIGHT, IMG_WIDTH, 1)

    pred = model.predict(image, verbose=0)[0]

    class_id = np.argmax(pred)
    confidence = np.max(pred)

    return CLASSES[class_id], confidence, pred

# =========================================================
# UI DRAW
# =========================================================

def draw_ui(frame, label, confidence, mode, fps):

    h, w = frame.shape[:2]

    # HEADER
    cv2.rectangle(frame, (0, 0), (w, 70), (20, 20, 20), -1)

    cv2.putText(
        frame,
        "SIBI AI DETECTION",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    # SIDE PANEL
    cv2.rectangle(frame, (20, 90), (360, 260), (30, 30, 30), -1)

    info = [
        f"Mode: LIVE PREDICTION",
        f"Prediction: {label}",
        f"Confidence: {confidence*100:.2f}%",
        f"FPS: {fps:.2f}"
    ]

    y = 130
    for t in info:
        cv2.putText(frame, t, (40, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        y += 35

    # WARNING LEVEL COLOR
    color = (0, 255, 0) if confidence > 0.8 else (0, 165, 255)

    cv2.putText(
        frame,
        "REAL TIME MODE",
        (w - 250, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )

# =========================================================
# MAIN LOOP
# =========================================================

def main():

    global bg

    cap = cv2.VideoCapture(0)

    num_frames = 0
    prev_time = 0

    print("[INFO] Starting camera...")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = imutils.resize(frame, width=900)
        frame = cv2.flip(frame, 1)

        clone = frame.copy()

        roi = frame[TOP:BOTTOM, RIGHT:LEFT]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # FPS
        curr_time = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (curr_time - prev_time)
        prev_time = curr_time

        # background init
        if num_frames < 30:
            run_avg(gray)
            cv2.putText(clone, "Calibrating Background...", (420, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:

            hand = segment(gray)

            if hand is not None:

                thresh, seg = hand

                cv2.drawContours(
                    clone,
                    [seg + (RIGHT, TOP)],
                    -1,
                    (0, 0, 255),
                    2
                )

                # PREDICTION REALTIME
                color_roi = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

                label, conf, _ = predict(color_roi)

                draw_ui(clone, label, conf, "LIVE", fps)

                cv2.imshow("Threshold", thresh)

        # ROI BOX
        cv2.rectangle(clone, (RIGHT, TOP), (LEFT, BOTTOM), (0, 255, 0), 2)

        cv2.imshow("SIBI AI (Sarip Hidayat) - Real Time Detection", clone)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        num_frames += 1

    cap.release()
    cv2.destroyAllWindows()

# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()