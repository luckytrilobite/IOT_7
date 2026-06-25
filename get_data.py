import cv2
import time
import threading
import json
import requests
from collections import deque
from ultralytics import YOLO
import paho.mqtt.client as mqtt
import ssl




# =========================
# CONFIG
# =========================
BROKER = "f9c1e85ec144455c9671301c943e8b29.s1.eu.hivemq.cloud"
TOPIC_SKELETON = "yolo/skeleton"
TOPIC_LSTM = "LSTM/errorpose"
UPLOAD_URL = "http://127.0.0.1:8004/get_video"

BUFFER_SEC = 5

# =========================
# SHARED FRAME QUEUE
# =========================
frame_queue = deque(maxlen=30)   # latest frames

# =========================
# BUFFER FOR VIDEO THREAD
# =========================
frame_buffer = deque()
recording = False
recorded_frames = []

lock = threading.Lock()

# =========================
# MODEL + MQTT
# =========================
model = YOLO("yolov8n-pose.pt")

client = mqtt.Client()

client.tls_set(cert_reqs=ssl.CERT_REQUIRED)
client.tls_insecure_set(False)
client.connect(BROKER, 8883, 60)

# =========================
# THREAD 1: CAMERA PRODUCER
# =========================
def camera_thread():
    cap = cv2.VideoCapture(0)
    print("Camera thread started")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        ts = time.time()

        # push to shared queue
        frame_queue.append((ts, frame.copy()))

        # optional display
        if cv2.waitKey(1) == 27:
            break

# =========================
# THREAD 2: YOLO CONSUMER + MQTT
# =========================
def yolo_thread():
    frame_id = 0
    print("YOLO thread started")

    while True:
        if len(frame_queue) == 0:
            continue

        ts, frame = frame_queue[-1]
        frame_id += 1

        results = model(frame, verbose=False)

        skeleton = None

        for r in results:
            if r.keypoints is None:
                continue

            kpts = r.keypoints.xy.cpu().numpy()

            if len(kpts) > 0:
                skeleton = kpts[0].reshape(-1).tolist()
                break

        if skeleton is None:
            skeleton = [0.0] * 34

        payload = {
            "frame_id": frame_id,
            "timestamp": ts,
            "skeleton": skeleton
        }

        client.publish(TOPIC_SKELETON, json.dumps(payload))

# =========================
# THREAD 3: BUFFER + VIDEO + ANOMALY
# =========================
def upload_video(frames):
    if len(frames) == 0:
        return

    frames = [f for _, f in frames]

    h, w = frames[0].shape[:2]

    out = cv2.VideoWriter(
        "anomaly.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        20,
        (w, h)
    )

    for f in frames:
        out.write(f)

    out.release()

    try:
        files = {"file": open("anomaly.mp4", "rb")}
        requests.post(UPLOAD_URL, files=files)
        print("UPLOAD DONE")
    except Exception as e:
        print("upload failed:", e)


def buffer_thread():
    global recording, frame_buffer, recorded_frames

    print("Buffer thread started")

    def on_message(client, userdata, msg):
        global recording, recorded_frames

        data = json.loads(msg.payload.decode())
        flag = data.get("errorpose", False)

        # =========================
        # START
        # =========================
        if flag and not recording:
            print("START RECORD")

            recording = True
            recorded_frames = list(frame_buffer)

        # =========================
        # STOP
        # =========================
        elif not flag and recording:
            print("STOP RECORD")

            recording = False
            upload_video(recorded_frames)

    client.subscribe(TOPIC_LSTM)
    client.on_message = on_message
    client.loop_start()

    while True:
        if len(frame_queue) == 0:
            continue

        ts, frame = frame_queue[-1]

        # =========================
        # 5-sec buffer
        # =========================
        frame_buffer.append((ts, frame.copy()))

        while frame_buffer and ts - frame_buffer[0][0] > BUFFER_SEC:
            frame_buffer.popleft()

        # =========================
        # record during anomaly
        # =========================
        if recording:
            recorded_frames.append((ts, frame.copy()))

# =========================
# MAIN
# =========================
if __name__ == "__main__":

    t1 = threading.Thread(target=camera_thread)
    t2 = threading.Thread(target=yolo_thread)
    t3 = threading.Thread(target=buffer_thread)

    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()