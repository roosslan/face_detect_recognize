import vlc
import time

# Строгие аргументы для исправления дедлока HEVC и сетевых потерь
vlc_args = [
    '--rtsp-tcp',                  # ТРЕБУЕТСЯ: переключает транспорт с UDP на TCP (убирает потерю опорных кадров)
    '--avcodec-hw=none',           # ТРЕБУЕТСЯ: отключает сбойный D3D11VA на ноутбучной RTX 4050
    '--avcodec-threads=4',         # Задействует 4 потока процессора для плавного софтового HEVC декодирования
    '--network-caching=300',       # Оптимальный буфер в мс (не 0, чтобы не было deadlock, но и не 3000)
    '--clock-jitter=0',            # Убирает рассинхронизацию времени пакетов
    '--clock-synchro=0',           # Отключает жесткую синхронизацию звука и видео ( приоритет скорости )
    '--drop-late-frames',          # Принудительно выкидывать отстающие кадры, не копить задержку
    '--skip-frames'                # Пропускать промежуточные кадры при нехватке мощности
]

instance = vlc.Instance(*vlc_args)
player = instance.media_player_new()

media = instance.media_new("rtsp://192.168.88.88:554/1")
player.set_media(media)
player.play()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    player.stop()


"""
import os
import cv2
import face_recognition
import numpy as np
import constconf
import main

import cv2
import threading
import time


class RTSPStream:
    def __init__(self, url):
        # Отключаем буфер FFmpeg (для OpenCV с поддержкой FFmpeg)
        # 0 означает, что буфер отключен, отдавать кадр сразу
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)

        self.ret, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.cap.isOpened():
                break
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame  # Здесь всегда самый свежий кадр
            time.sleep(0.01)  # Не перегружаем процессор

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()


# Использование:
rtsp_url = "rtsp://192.168.88.88:554/0"
stream = RTSPStream(rtsp_url).start()

while True:
    frame = stream.read()
    if frame is not None:
        cv2.imshow('No Lag Stream', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        stream.stop()
        break

cv2.destroyAllWindows()

""

known_face_encodings = []
known_face_names = []
faces_dir = "faces"  # директория с фото (имя файла = имя person)

if os.path.exists(faces_dir):
    for filename in os.listdir(faces_dir):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(faces_dir, filename)
            image = face_recognition.load_image_file(path)
            # Получаем цифровой "отпечаток" лица
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(os.path.splitext(filename)[0])

print(f"Загружено лице людей: {len(known_face_names)}")

video_capture = cv2.VideoCapture(constconf.RTSP_URL)

if not video_capture.isOpened():
    print("Ошибка: Не удалось подключиться к RTSP-потоку")
    exit()

# Переменная для пропуска кадров (оптимизация производительности)
process_this_frame = True

while True:
    # Захватываем один кадр видео
    ret, frame = video_capture.read()
    if not ret:
        print("Потерян сигнал RTSP")
        break

    # Обрабатываем только каждый второй кадр для экономии ресурсов
    if process_this_frame:
        # Уменьшаем кадр в 4 раза для ускорения обработки face_recognition
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # Конвертируем BGR (OpenCV) в RGB (face_recognition)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Находим все лица и их отпечатки на текущем кадре
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Проверяем, совпадает ли лицо с кем-то из известных
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"  # Неизвестный человек

            # Альтернативный вариант: ищем максимальное совпадение
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # --- 3. ОТРИСОВКА РЕЗУЛЬТАТОВ ---
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Возвращаем координаты к исходному размеру экрана (умножаем на 4)
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Рисуем рамку вокруг лица
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Выводим имя под рамкой
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

    # Отображаем окно с видео
    cv2.imshow('RTSP Face Recognition', frame)

    # Выход из программы при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
video_capture.release()
cv2.destroyAllWindows()



"""