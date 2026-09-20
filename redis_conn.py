import os
import sys
import time
from datetime import datetime
import constconf
import cv2
import face_recognition
import numpy as np
import redis

def get_redis():
    """ подключение к redis с проверкой доступности (ping) """
    client = redis.Redis(
        host = constconf.REDIS_HOST,
        port = constconf.REDIS_PORT,
        db = constconf.REDIS_DB,
        decode_responses = True,
        socket_connect_timeout = 3,
    )
    client.ping()  # кинет эксепшн, если сервер недоступен
    return client


def log_face_event(client, name, when=None):
    """
    Записывает событие обнаружения лица в Redis Stream.

    Запись вида: time = "2026-09-12 13:21:00", name = "rasa".
    Возвращает ID добавленной записи.
    """
    when = when or datetime.now()
    return client.xadd(
        constconf.STREAM_KEY,
        {
            "time": when.strftime("%Y-%m-%d %H:%M:%S"),
            "name": name,
        },
        maxlen = constconf.STREAM_MAXLEN,
        approximate=True,
    )


def read_events(client, count=50):
    """Возвращает последние `count` событий (новые сверху)."""
    return client.xrevrange(constconf.STREAM_KEY, count=count)


# --- 3. ЗАГРУЗКА ИЗВЕСТНЫХ ЛИЦ ---
def load_known_faces():
    """Читает папку faces: имя файла (без расширения) = имя человека."""
    encodings, names = [], []
    if os.path.exists(constconf.FACES_DIR):
        for filename in os.listdir(constconf.FACES_DIR):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(constconf.FACES_DIR, filename)
                image = face_recognition.load_image_file(path)
                face_encs = face_recognition.face_encodings(image)
                if face_encs:
                    encodings.append(face_encs[0])
                    names.append(os.path.splitext(filename)[0])
    return encodings, names

