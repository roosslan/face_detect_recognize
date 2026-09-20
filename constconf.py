RTSP_URL = "rtsp://10.X.XX.XX:554/2"
REDIS_HOST = "127.0.0.1"
REDIS_PORT = 6379
REDIS_DB = 0

FACES_DIR = "faces"
STREAM_KEY = "face_events"   # ключ Redis Stream, куда пишутся события
STREAM_MAXLEN = 10_000       # хранить не более ~10 000 последних записей
PROCESS_EVERY_NTH_FRAME = 2  # распознавать каждый N-й кадр (оптимизация)
LOG_COOLDOWN_SEC = 10.0      # одного человека записываем не чаще, чем раз в N секунд
LOG_UNKNOWN = False          # записывать ли неопознанных ("Unknown")
