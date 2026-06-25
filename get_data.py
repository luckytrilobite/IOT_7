import cv2
import time
import os
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
STREAM_URL = "rtmp://172.20.10.8/live/stream"

SEQ_LEN = 30
FPS = 20

PRE_SEC = 5
POST_SEC = 3

PRE_BUF_LEN = PRE_SEC * FPS
POST_BUF_LEN = POST_SEC * FPS

SAVE_DIR = "./videos"
os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# LOAD PARAMS
# =========================
mean = np.load("./LSTM/norm_mean.npy")
std = np.load("./LSTM/norm_std.npy")
threshold = float(np.load("./LSTM/threshold.npy"))

# =========================
# MODEL
# =========================
class LSTMAE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        h = h[-1]
        h = h.unsqueeze(1).repeat(1, x.size(1), 1)
        return self.decoder(h)[0]

model = LSTMAE(34, 64)
model.load_state_dict(torch.load("./LSTM/lstm_ae.pth", map_location="cpu"))
model.eval()

# =========================
# YOLO
# =========================
yolo = YOLO("yolov8n-pose.pt").to("cpu")

# =========================
# BUFFER
# =========================
frame_buffer = deque(maxlen=PRE_BUF_LEN)
seq_buffer = deque(maxlen=SEQ_LEN)

# =========================
# STATE
# =========================
recording = False
post_counter = 0
video_id = 0
record_buffer = []

# =========================
# SMOOTHING STATE
# =========================
person_counter = 0
PERSON_HOLD = 5

ema_error = 0.0
alpha = 0.2

is_anomaly = False

# =========================
# SAVE VIDEO
# =========================
def save_video(frames, vid):
    if len(frames) == 0:
        return

    h, w = frames[0].shape[:2]
    path = f"{SAVE_DIR}/event_{vid}.mp4"

    out = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (w, h)
    )

    for f in frames:
        out.write(f)

    out.release()
    print(f"[SAVE] {path}")

# =========================
# SAFE SKELETON
# =========================
def get_skeleton(result):
    if result.keypoints is None:
        return None

    kpts = result.keypoints.xy.cpu().numpy()
    if kpts is None or len(kpts) == 0:
        return None

    sk = kpts[0].reshape(-1)
    if len(sk) != 34:
        return None

    return sk

# =========================
# STREAM
# =========================
cap = cv2.VideoCapture(STREAM_URL)

if not cap.isOpened():
    print("Cannot open stream")
    exit()

print("Stream opened")

# =========================
# LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # =========================
    # YOLO
    # =========================
    results = yolo(frame, verbose=False)

    skeleton = None
    detected = False

    for r in results:
        skeleton = get_skeleton(r)
        if skeleton is not None:
            detected = True
            break

    # =========================
    # PERSON FILTER
    # =========================
    if detected:
        person_counter = 0
    else:
        person_counter += 1

    is_person = person_counter < PERSON_HOLD

    if not is_person or skeleton is None:
        skeleton = np.zeros(34, dtype=np.float32)

    skeleton = np.asarray(skeleton, dtype=np.float32)

    # =========================
    # NORMALIZE
    # =========================
    skeleton_norm = (skeleton - mean) / std
    seq_buffer.append(skeleton_norm)

    # =========================
    # LSTM AE
    # =========================
    error = 0.0

    if len(seq_buffer) == SEQ_LEN:
        x = torch.tensor(np.array(seq_buffer), dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            recon = model(x)
            error = ((x - recon) ** 2).mean().item()

    # =========================
    # EMA SMOOTH
    # =========================
    ema_error = alpha * error + (1 - alpha) * ema_error

    # =========================
    # HYSTERESIS (ONLY ONE SOURCE OF TRUTH)
    # =========================
    ON_TH = threshold
    OFF_TH = threshold * 0.7

    if ema_error > ON_TH:
        is_anomaly = True
    elif ema_error < OFF_TH:
        is_anomaly = False

    if not is_person:
        is_anomaly = False
        ema_error = 0.0   # 🔥 防止殘留記憶污染下一段
        seq_buffer.clear()

    # =========================
    # FRAME BUFFER
    # =========================
    frame_buffer.append(frame.copy())

    # =========================
    # EVENT START
    # =========================
    if is_anomaly and not recording:
        print("[EVENT START]")
        recording = True
        record_buffer = list(frame_buffer)

    # =========================
    # RECORDING
    # =========================
    if recording:
        record_buffer.append(frame.copy())

        if not is_anomaly:
            post_counter += 1
        else:
            post_counter = 0

        if post_counter >= POST_BUF_LEN:
            print("[EVENT END]")
            save_video(record_buffer, video_id)

            video_id += 1
            recording = False
            post_counter = 0

    # =========================
    # DEBUG
    # =========================
    print({
        "is_person": is_person,
        "is_anomaly": is_anomaly,
        "error": float(error),
        "ema_error": float(ema_error)
    })

    cv2.imshow("stream", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()