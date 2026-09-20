"""Command-line application: RTSP -> face recognition -> Redis, with a live preview."""

import argparse
import logging
import sys
import time

from facerec import events
from facerec.capture import LatestFrameReader, make_rtsp_capture
from facerec.config import Config, ConfigError, load_config, redact_url
from facerec.faces import load_known_faces
from facerec.recognizer import UNKNOWN, DlibEngine, FaceRecognizer, RecognitionWorker

log = logging.getLogger("facerec")

WINDOW = "RTSP Face Recognition -> Redis"
NO_SIGNAL_AFTER_SEC = 3.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="facerec", description=__doc__)
    parser.add_argument("--config", default="config.toml", help="path to the TOML config")
    parser.add_argument("--headless", action="store_true", help="run without the preview window")
    parser.add_argument(
        "--list",
        nargs="?",
        const=20,
        type=int,
        metavar="N",
        help="print the last N events from Redis (default 20) and exit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
    )

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        log.error("%s", exc)
        return 2

    try:
        client = events.connect_redis(config.redis)
    except Exception as exc:
        log.error("Redis is unavailable (%s:%s): %s", config.redis.host, config.redis.port, exc)
        return 1

    if args.list is not None:
        return print_events(client, config, args.list)

    known = load_known_faces(config.recognition.faces_dir)
    if not len(known):
        log.warning(
            "No faces loaded from '%s': everybody will be 'Unknown'", config.recognition.faces_dir
        )

    recognizer = FaceRecognizer(
        DlibEngine(
            config.recognition.detect_scale,
            config.recognition.model,
            config.recognition.upsample,
        ),
        known,
        config.recognition.tolerance,
    )
    event_logger = events.EventLogger(
        client,
        config.redis.stream_key,
        config.redis.stream_maxlen,
        config.events.log_cooldown_sec,
        config.events.log_unknown,
    )
    cam = config.camera
    log.info("Camera: %s (%s)", redact_url(cam.rtsp_url), cam.transport)
    reader = LatestFrameReader(
        lambda: make_rtsp_capture(cam.rtsp_url, cam.transport, cam.timeout_sec),
        cam.reconnect_delay_sec,
    )
    worker = RecognitionWorker(
        reader, recognizer, event_logger.handle, config.recognition.min_interval_sec
    )

    reader.start()
    worker.start()
    try:
        if args.headless:
            run_headless()
        else:
            run_viewer(reader, worker)
    except KeyboardInterrupt:
        pass
    finally:
        worker.stop()
        reader.stop()
    return 0


def print_events(client, config: Config, count: int) -> int:
    entries = events.read_events(client, config.redis.stream_key, count)
    if not entries:
        print(f"No events in '{config.redis.stream_key}' yet")
    for entry_id, fields in entries:
        print(f"{fields.get('time', '?')}  {fields.get('name', '?')}  (id: {entry_id})")
    return 0


def run_headless() -> None:
    log.info("Running headless, press Ctrl+C to stop")
    while True:
        time.sleep(1.0)


def run_viewer(reader: LatestFrameReader, worker: RecognitionWorker) -> None:
    import cv2

    last_seq = 0
    try:
        while True:
            snap = reader.wait_for_new(last_seq, timeout=0.1) or reader.latest()
            if snap is None:
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break
                continue
            last_seq = snap.seq
            # the worker reads the shared frame from another thread, so draw on a copy
            canvas = snap.frame.copy()
            draw_detections(canvas, worker.latest_detections())
            if time.monotonic() - snap.timestamp > NO_SIGNAL_AFTER_SEC:
                cv2.putText(
                    canvas, "NO SIGNAL", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 0, 255), 2
                )
            cv2.imshow(WINDOW, canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break  # window closed with the X button
    finally:
        cv2.destroyAllWindows()


def draw_detections(canvas, detections) -> None:
    import cv2

    for det in detections:
        top, right, bottom, left = det.box
        color = (0, 0, 255) if det.name == UNKNOWN else (0, 255, 0)
        cv2.rectangle(canvas, (left, top), (right, bottom), color, 2)
        cv2.rectangle(canvas, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(
            canvas,
            det.name,
            (left + 6, bottom - 6),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (255, 255, 255),
            1,
        )


if __name__ == "__main__":
    sys.exit(main())
