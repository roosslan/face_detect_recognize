"""Command-line application: RTSP -> face recognition -> Redis, with a live preview."""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from facerec import events
from facerec.capture import LatestFrameReader, make_rtsp_capture
from facerec.config import Config, ConfigError, load_config, redact_url
from facerec.faces import load_known_faces
from facerec.motion import MotionDetector
from facerec.overlay import annotated, draw_detections
from facerec.recognizer import DlibEngine, FaceRecognizer, RecognitionWorker
from facerec.snapshots import MotionSnapshotter, MotionWorker, SnapshotWriter

log = logging.getLogger("facerec")

WINDOW = "RTSP Face Recognition -> Redis"
NO_SIGNAL_AFTER_SEC = 3.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="facerec", description=__doc__)
    parser.add_argument(
        "--config",
        help="path to the TOML config (default: config.toml next to the exe, or in the "
        "current folder when run as a script)",
    )
    parser.add_argument("--headless", action="store_true", help="run without the preview window")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="check that all libraries and dlib models load (useful for a freshly built exe)",
    )
    parser.add_argument(
        "--list",
        nargs="?",
        const=20,
        type=int,
        metavar="N",
        help="print the last N events from Redis (default 20) and exit",
    )
    return parser


def frozen_app_dir() -> Path | None:
    """Folder of the exe when running as a PyInstaller build, None when run as a script."""
    return Path(sys.executable).parent if getattr(sys, "frozen", False) else None


def run_self_test() -> None:
    """Load everything the program needs and run detection on a blank image.

    Fails if a library or a dlib model file is missing, which is what can go wrong in a
    freshly built exe: it starts fine and would only fail on the first camera frame."""
    import cv2
    import numpy
    import redis

    log.info("cv2 %s, numpy %s, redis %s", cv2.__version__, numpy.__version__, redis.__version__)
    DlibEngine().detect(numpy.zeros((120, 160, 3), numpy.uint8))  # loads dlib and its models


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
    )

    if args.self_test:
        try:
            run_self_test()
        except Exception:
            log.exception("Self-test failed")
            return 1
        log.info("Self-test passed")
        return 0

    # The exe keeps config.toml, faces/ and capture/ next to itself, whatever folder it was
    # started from (a shortcut or a scheduled task may use any). An explicit --config is taken
    # relative to where the user typed it.
    exe_dir = frozen_app_dir()
    if args.config:
        config_path = Path(args.config).resolve()
    else:
        config_path = (exe_dir or Path.cwd()) / "config.toml"
    if exe_dir:
        os.chdir(exe_dir)

    try:
        config = load_config(config_path)
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
    snaps = config.snapshots
    # Shared with the motion snapshotter below, so a Redis event and a motion trigger in the
    # same minute get " (2)" instead of one silently overwriting the other.
    snapshot_writer = SnapshotWriter(snaps.directory) if snaps.enabled else None
    snapshot_annotate = annotated if snaps.enabled and snaps.annotate else None
    event_logger = events.EventLogger(
        client,
        config.redis.stream_key,
        config.redis.stream_maxlen,
        config.events.log_cooldown_sec,
        config.events.log_unknown,
        snapshots=snapshot_writer,
        annotate=snapshot_annotate,
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

    workers = [worker]
    if snapshot_writer is not None:
        snapshotter = MotionSnapshotter(
            MotionDetector(snaps.motion_delta, snaps.motion_area),
            snapshot_writer,
            worker.latest_detections,
            snaps.cooldown_sec,
            snaps.settle_sec,
            annotate=snapshot_annotate,
        )
        workers.append(MotionWorker(reader, snapshotter))
        log.info("Snapshots on motion, and on every Redis event -> '%s'", snaps.directory)

    reader.start()
    for w in workers:
        w.start()
    try:
        if args.headless:
            run_headless()
        else:
            run_viewer(reader, worker)
    except KeyboardInterrupt:
        pass
    finally:
        for w in workers:
            w.stop()
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


def _window_is_open(cv2) -> bool:
    """False once the window is gone. Closing it with the X button can, on some OpenCV/Qt
    builds (seen on Linux), tear the window down immediately instead of just hiding it, so the
    very next property check raises `cv2.error` ("NULL guiReceiver") rather than returning < 1
    like it does everywhere else; that must count as "closed" too, not crash the program."""
    try:
        return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


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
            if not _window_is_open(cv2):
                break  # window closed with the X button
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
