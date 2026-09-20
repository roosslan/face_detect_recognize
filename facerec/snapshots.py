"""Snapshots taken when something moves in front of the camera."""

import logging
import re
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from facerec.capture import LatestFrameReader
from facerec.motion import MotionDetector
from facerec.recognizer import UNKNOWN, Detection

log = logging.getLogger(__name__)

_INVALID_IN_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def snapshot_stem(when: datetime, names: list[str]) -> str:
    """`dd-mm-yyyy-hh-mm` plus `-person_name` when somebody was recognized.

    Several people are joined with '+'."""
    stem = when.strftime("%d-%m-%Y-%H-%M")
    if names:
        stem += "-" + "+".join(_INVALID_IN_FILENAME.sub("_", name) for name in sorted(names))
    return stem


def jpeg_encode(frame: Any, quality: int = 90) -> bytes:
    import cv2

    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("cannot encode the frame as JPEG")
    return buffer.tobytes()


class SnapshotWriter:
    def __init__(
        self,
        directory: str | Path,
        encode: Callable[[Any], bytes] = jpeg_encode,
        extension: str = ".jpg",
    ):
        self._directory = Path(directory)
        self._encode = encode
        self._extension = extension

    def save(self, frame: Any, names: list[str], when: datetime) -> Path:
        """Write the frame; the name has only minute precision, so a second snapshot within
        the same minute gets ' (2)', ' (3)'... instead of overwriting the first one."""
        data = self._encode(frame)  # encode first: a failure must not leave an empty file
        self._directory.mkdir(parents=True, exist_ok=True)
        stem = snapshot_stem(when, names)
        number = 1
        while True:
            suffix = "" if number == 1 else f" ({number})"
            path = self._directory / f"{stem}{suffix}{self._extension}"
            try:
                with path.open("xb") as fh:  # "x": never overwrite an existing file
                    fh.write(data)
                return path
            except FileExistsError:
                number += 1


class MotionSnapshotter:
    """Takes a snapshot when motion starts.

    Recognition needs a moment (a person has to enter the frame and turn to the camera), so
    after motion starts it waits up to `settle` seconds for a known person to be recognized:
    the snapshot is taken as soon as somebody is named, or at the end of that window without
    a name. After a snapshot nothing is taken for `cooldown` seconds.
    """

    def __init__(
        self,
        detector: MotionDetector,
        writer: SnapshotWriter,
        get_detections: Callable[[], list[Detection]],
        cooldown: float = 10.0,
        settle: float = 1.5,
        annotate: Callable[[Any, list[Detection]], Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = datetime.now,
    ):
        self._detector = detector
        self._writer = writer
        self._get_detections = get_detections
        self._cooldown = cooldown
        self._settle = settle
        self._annotate = annotate
        self._clock = clock
        self._now = now
        self._deadline: float | None = None  # set while waiting for a face after motion started
        self._last_saved: float | None = None

    def process(self, frame: Any) -> Path | None:
        """Handle the next frame; returns the path if a snapshot was written."""
        t = self._clock()
        moving = self._detector.update(frame)
        if self._deadline is None:
            cooled_down = self._last_saved is None or t - self._last_saved >= self._cooldown
            if not (moving and cooled_down):
                return None
            self._deadline = t + self._settle

        detections = self._get_detections()
        names = sorted({d.name for d in detections if d.name != UNKNOWN})
        if not names and t < self._deadline:
            return None

        # whatever happens next, do not retry on every frame (e.g. when the disk is full)
        self._deadline, self._last_saved = None, t
        image = self._annotate(frame, detections) if self._annotate else frame
        return self._writer.save(image, names, self._now())


class MotionWorker:
    """Runs the snapshotter on the newest frames in its own thread."""

    def __init__(
        self,
        reader: LatestFrameReader,
        snapshotter: MotionSnapshotter,
        min_interval: float = 0.1,
    ):
        self._reader = reader
        self._snapshotter = snapshotter
        self._min_interval = min_interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="motion", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._thread.join(timeout)

    def _run(self) -> None:
        last_seq = 0
        while not self._stop.is_set():
            snap = self._reader.wait_for_new(last_seq, timeout=0.5)
            if snap is None:
                continue
            last_seq = snap.seq
            started = time.monotonic()
            try:
                path = self._snapshotter.process(snap.frame)
            except Exception:
                log.exception("Snapshot failed")
                self._stop.wait(1.0)
                continue
            if path is not None:
                log.info("Snapshot: %s", path)
            remaining = self._min_interval - (time.monotonic() - started)
            if remaining > 0:
                self._stop.wait(remaining)
