import cv2
import subprocess

RTMP_URL = "rtmp://172.20.10.8/live/stream"

cap = cv2.VideoCapture(0)

w = 640
h = 480
fps = 15

cmd = [
    "ffmpeg",
    "-y",
    "-f", "rawvideo",
    "-pix_fmt", "bgr24",
    "-s", f"{w}x{h}",
    "-r", str(fps),
    "-i", "-",
    "-c:v", "libx264",
    "-preset", "ultrafast",
    "-tune", "zerolatency",
    "-f", "flv",
    RTMP_URL
]

proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (w, h))
    proc.stdin.write(frame.tobytes())
