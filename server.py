from fastapi import FastAPI, UploadFile, File
import os
import pymysql
from datetime import datetime

app = FastAPI()

# =========================
# DB CONFIG
# =========================
db = pymysql.connect(
    host="127.0.0.1",
    user="user",
    password="userpass",
    database="testdb",
    autocommit=True
)

cursor = db.cursor()

# =========================
# STORAGE PATH
# =========================
SAVE_DIR = "/home/hsieh/SQL/save_videos"
os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# TIME ID
# =========================
def gen_event_id():
    return datetime.now().strftime("%y%m%d%H%M%S")


# =========================================================
# API 1: /get_video
# =========================================================
@app.post("/get_video")
async def get_video(file: UploadFile = File(...)):
    event_id = gen_event_id()

    filename = f"{event_id}_{file.filename}"
    filepath = os.path.join(SAVE_DIR, filename)

    # save file
    with open(filepath, "wb") as f:
        f.write(await file.read())

    # insert DB
    sql = """
        INSERT INTO anomaly_video (event_id, filename, filepath)
        VALUES (%s, %s, %s)
    """
    cursor.execute(sql, (event_id, filename, filepath))

    return {
        "status": "ok",
        "event_id": event_id,
        "file": filename
    }


# =========================================================
# API 2: /get_file (同功能但獨立 endpoint)
# =========================================================
@app.post("/get_file")
async def get_file(file: UploadFile = File(...)):
    event_id = gen_event_id()

    filename = f"{event_id}_{file.filename}"
    filepath = os.path.join(SAVE_DIR, filename)

    with open(filepath, "wb") as f:
        f.write(await file.read())

    sql = """
        INSERT INTO anomaly_video (event_id, filename, filepath)
        VALUES (%s, %s, %s)
    """
    cursor.execute(sql, (event_id, filename, filepath))

    return {
        "status": "saved",
        "event_id": event_id,
        "path": filepath
    }