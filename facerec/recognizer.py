"""Face detection + recognition on frames, and the background worker running it."""

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from facerec.capture import LatestFrameReader
from facerec.faces import KnownFaces

log = logging.getLogger(__name__)

UNKNOWN = "Unknown"

Box = tuple[int, int, int, int]  # (top, right, bottom, left) in full-frame pixels


@dataclass(frozen=True)
class Detection:
    name: str
    box: Box
    distance: float | None = None


class FaceEngine(Protocol):
    def detect(self, frame_bgr: Any) -> list[tuple[Box, np.ndarray]]:
        """Find faces on a BGR frame; return (box, encoding) for each."""


class DlibEngine:
    """Detection and encoding through the `face_recognition` package (dlib)."""

    def __init__(self, scale: float = 0.75, model: str = "hog", upsample: int = 1):
        self._scale = scale
        self._model = model
        self._upsample = upsample

    def detect(self, frame_bgr: Any) -> list[tuple[Box, np.ndarray]]:
        import cv2
        import face_recognition

        # Detection is the expensive part and needs the face to be at least ~80 px (40 px with
        # upsample=1) in the image it runs on, so it works on a scaled copy of the frame.
        # Encoding always uses the full-resolution frame: a small face looks much more like
        # itself there than on the downscaled copy, which decides whether far faces match.
        if self._scale == 1:
            small = frame_bgr
        else:
            small = cv2.resize(frame_bgr, (0, 0), fx=self._scale, fy=self._scale)
        boxes_small = face_recognition.face_locations(
            cv2.cvtColor(small, cv2.COLOR_BGR2RGB),
            number_of_times_to_upsample=self._upsample,
            model=self._model,
        )
        if not boxes_small:
            return []
        inv = 1 / self._scale
        boxes = [tuple(int(v * inv) for v in box) for box in boxes_small]
        encodings = face_recognition.face_encodings(
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), boxes
        )
        return list(zip(boxes, encodings, strict=True))


class FaceRecognizer:
    def __init__(self, engine: FaceEngine, known: KnownFaces, tolerance: float = 0.6):
        self._engine = engine
        self._known = known
        self._tolerance = tolerance

    def identify(self, frame_bgr: Any) -> list[Detection]:
        detections = []
        for box, encoding in self._engine.detect(frame_bgr):
            name, distance = self._known.match(encoding, self._tolerance)
            detections.append(Detection(name or UNKNOWN, box, distance))
        return detections


class RecognitionWorker:
    """Runs recognition on the newest frame, as fast as it can, in its own thread."""

    def __init__(
        self,
        reader: LatestFrameReader,
        recognizer: FaceRecognizer,
        on_detections: Callable[[list[Detection], Any], None] | None = None,
        min_interval: float = 0.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._reader = reader
        self._recognizer = recognizer
        self._on_detections = on_detections
        self._min_interval = min_interval
        self._clock = clock
        self._lock = threading.Lock()
        self._result: tuple[list[Detection], float] = ([], float("-inf"))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="recognition", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._thread.join(timeout)

    def latest_detections(self, max_age: float = 1.0) -> list[Detection]:
        """Last result, or [] if it is older than `max_age` seconds (a stale box would
        otherwise hang on screen after the recognition stalls)."""
        with self._lock:
            detections, stamp = self._result
        return detections if self._clock() - stamp <= max_age else []

    def _run(self) -> None:
        last_seq = 0
        while not self._stop.is_set():
            snap = self._reader.wait_for_new(last_seq, timeout=0.5)
            if snap is None:
                continue
            last_seq = snap.seq
            started = self._clock()
            try:
                detections = self._recognizer.identify(snap.frame)
            except Exception:
                log.exception("Recognition failed")
                self._stop.wait(1.0)
                continue
            with self._lock:
                self._result = (detections, self._clock())
            if self._on_detections is not None:
                try:
                    self._on_detections(detections, snap.frame)
                except Exception:
                    log.exception("Detection handler failed")
            remaining = self._min_interval - (self._clock() - started)
            if remaining > 0:
                self._stop.wait(remaining)
