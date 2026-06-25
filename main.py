import json
import numpy as np
import torch
import torch.nn as nn
import paho.mqtt.client as mqtt
from collections import deque

# =========================
# CONFIG
# =========================
BROKER = "127.0.0.1"

TOPIC_IN = "yolo/skeleton"
TOPIC_OUT = "LSTM/errorpose"

SEQ_LEN = 30
INPUT_DIM = 34
HIDDEN_DIM = 64

device = "cpu"

# =========================
# LOAD normalization + threshold
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

        h = h[-1]  # (B, H)
        h_repeat = h.unsqueeze(1).repeat(1, x.size(1), 1)

        out, _ = self.decoder(h_repeat)
        return out

model = LSTMAE(INPUT_DIM, HIDDEN_DIM).to(device)
model.load_state_dict(torch.load("./LSTM/lstm_ae.pth", map_location=device))
model.eval()

# =========================
# BUFFER
# =========================
buffer = deque(maxlen=SEQ_LEN)

# =========================
# MQTT
# =========================
client = mqtt.Client()
client.connect(BROKER, 1883, 60)

# =========================
# CALLBACK
# =========================
def on_message(client, userdata, msg):
    global buffer

    data = json.loads(msg.payload.decode())
    skeleton = np.array(data["skeleton"], dtype=np.float32)

    # =========================
    # normalize
    # =========================
    skeleton = (skeleton - mean) / std

    buffer.append(skeleton)

    # wait until full sequence
    if len(buffer) < SEQ_LEN:
        return

    seq = np.array(buffer)  # (T, F)

    x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        recon = model(x)
        error = torch.mean((x - recon) ** 2).item()

    is_anomaly = error > threshold

    payload = {
        "errorpose": bool(is_anomaly),
        "score": float(error)
    }

    client.publish(TOPIC_OUT, json.dumps(payload))


# =========================
# START
# =========================
client.subscribe(TOPIC_IN)
client.on_message = on_message

print("LSTM AE inference running...")
client.loop_forever()