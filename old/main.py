import os
import sys
import time
from datetime import datetime
import constconf
import cv2
import face_recognition
import numpy as np
import redis
import redis_conn

# --- 4. ОСНОВНОЙ ЦИКЛ ---
def main():
    # Режим просмотра последних событий (без камеры)
    if "--list" in sys.argv:
        r = redis_conn.get_redis()
        events = redis_conn.read_events(r, count=20)
        if not events:
            print(f"В ключе '{constconf.STREAM_KEY}' пока нет записей")
        for entry_id, fields in events:
            print(f"{fields.get('time', '?')}  {fields.get('name', '?')}  (id: {entry_id})")
        return

    # Подключаемся к Redis
    try:
        r = redis_conn.get_redis()
        print(f"Redis подключен: {constconf.REDIS_HOST}:{constconf.REDIS_PORT}, ключ '{constconf.STREAM_KEY}'")
    except Exception as e:
        print(f"Ошибка: Redis недоступен ({e})")
        return

    # Загружаем известные лица
    known_face_encodings, known_face_names = redis_conn.load_known_faces()
    print(f"Загружено профилей людей: {len(known_face_names)}")
    if not known_face_names:
        print("Внимание: в папке 'faces' не найдено ни одного лица")

    # Подключаемся к RTSP-камере
    video_capture = cv2.VideoCapture(constconf.RTSP_URL)
    if not video_capture.isOpened():
        print("Ошибка: Не удалось подключиться к RTSP-потоку")
        return

    frame_count = 0
    face_locations, face_names = [], []
    last_logged = {}  # name -> время (monotonic) последней записи в Redis

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Потерян сигнал RTSP")
            break

        frame_count += 1

        # Распознаём только каждый N-й кадр (экономия ресурсов)
        if frame_count % constconf.PROCESS_EVERY_NTH_FRAME == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(
                    known_face_encodings, face_encoding, tolerance=0.6
                )
                name = "Unknown"
                face_distances = face_recognition.face_distance(
                    known_face_encodings, face_encoding
                )
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                face_names.append(name)

                # --- ЗАПИСЬ СОБЫТИЯ ОБНАРУЖЕНИЯ В REDIS ---
                if name != "Unknown" or constconf.LOG_UNKNOWN:
                    now_mono = time.monotonic()
                    if now_mono - last_logged.get(name, 0.0) >= constconf.LOG_COOLDOWN_SEC:
                        try:
                            entry_id = log_face_event(r, name)
                            last_logged[name] = now_mono
                            print(
                                f"[REDIS] Событие записано: {name} "
                                f"@ {datetime.now():%Y-%m-%d %H:%M:%S} (id: {entry_id})"
                            )
                        except Exception as e:
                            print(f"[REDIS] Ошибка записи: {e}")

        # --- ОТРИСОВКА РЕЗУЛЬТАТОВ ---
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Возвращаем координаты к исходному размеру (кадр уменьшался в 4 раза)
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(
                frame, name, (left + 6, bottom - 6),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1,
            )

        cv2.imshow("RTSP Face Recognition -> Redis", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()