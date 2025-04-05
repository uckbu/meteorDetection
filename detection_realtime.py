import cv2
import numpy as np
import onnxruntime as ort
from collections import deque
import datetime
import os

# === CONFIGURATION ===
MODEL_PATH           = "yolov5s.onnx"   # or your meteor model
FRAME_WIDTH, FRAME_HEIGHT = 640, 640
FPS                  = 30
DETECTION_THRESHOLD  = 0.5  # confidence threshold
VIDEOS_DIR           = "videos"

os.makedirs(VIDEOS_DIR, exist_ok=True)

# === LOAD MODEL ===
providers = [
    'TensorrtExecutionProvider',
    'CUDAExecutionProvider',
    'CPUExecutionProvider'
]
session    = ort.InferenceSession(MODEL_PATH, providers=providers)
input_name = session.get_inputs()[0].name

def preprocess(frame):
    # resize, BGR→RGB, normalize, CHW, batch
    img = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)

def run_inference(frame):
    """
    Returns a list of detections: [(x1,y1,x2,y2,conf), ...]
    Assumes a YOLOv5 ONNX with raw preds [cx,cy,w,h,obj,cls0,cls1...].
    Applies simple decoding + NMS.
    """
    inp = preprocess(frame)
    preds = np.squeeze(session.run(None, {input_name: inp})[0])
    if preds.ndim == 1:
        preds = np.expand_dims(preds, axis=0)

    boxes      = []
    confidences = []

    for p in preds:
        obj_conf = float(p[4])
        if obj_conf < DETECTION_THRESHOLD:
            continue

        # get class confidence (if you only have one class you can skip)
        class_scores = p[5:]
        class_id     = int(np.argmax(class_scores))
        class_conf   = float(class_scores[class_id])
        conf         = obj_conf * class_conf
        if conf < DETECTION_THRESHOLD:
            continue

        # decode from center‐wh to x,y,w,h in pixels
        cx, cy, w, h = p[0], p[1], p[2], p[3]
        x = (cx - w/2.0) * FRAME_WIDTH
        y = (cy - h/2.0) * FRAME_HEIGHT
        w_pix = w * FRAME_WIDTH
        h_pix = h * FRAME_HEIGHT

        boxes.append([int(x), int(y), int(w_pix), int(h_pix)])
        confidences.append(conf)

    # run NMS
    indices = cv2.dnn.NMSBoxes(boxes, confidences,
                               DETECTION_THRESHOLD, 0.4)
    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w_pix, h_pix = boxes[i]
            x1, y1 = x, y
            x2, y2 = x + w_pix, y + h_pix
            results.append((x1, y1, x2, y2, confidences[i]))
    return results

def gstreamer_pipeline():
    return ("nvarguscamerasrc ! video/x-raw(memory:NVMM), width={0}, height={1}, "
            "format=NV12, framerate={2}/1 ! nvvidconv ! video/x-raw, width={0}, height={1}, "
            "format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
           ).format(FRAME_WIDTH, FRAME_HEIGHT, FPS)

# === INITIALIZE CAMERA & WINDOW ===
cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
if not cap.isOpened():
    print("Error: Unable to open camera.")
    exit()

cv2.namedWindow("Meteor Detection", cv2.WINDOW_AUTOSIZE)

print("[INFO] Starting live detection. Press 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # run model
        dets = run_inference(frame)

        # draw boxes
        for (x1, y1, x2, y2, conf) in dets:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = "Meteor: {0:.2f}".format(conf)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # show
        cv2.imshow("Meteor Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
