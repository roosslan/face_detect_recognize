"""RTSP capture that always exposes the newest frame.

Reading and processing run in different threads: the reader keeps draining the
stream, so the decoder buffer never fills up and latency does not accumulate,
no matter how slow face recognition is. Frames nobody asked for are dropped.
"""

import logging
import os
import threading
import time
from collections.abc import Callable
from typing import Any, NamedTuple

log = logging.getLogger(__name__)


class Snapshot(NamedTuple):
    frame: Any
    seq: int  # increases with every new frame, starts at 1
    timestamp: float  # time.monotonic() at the moment the frame was received


class LatestFrameReader:
    def __init__(
        self,
        open_capture: Callable[[], Any],
        reconnect_delay: float = 3.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        """`open_capture` returns an object with isOpened(), read() and release()
        (cv2.VideoCapture-compatible). It is called again after every lost connection."""
        self._open_capture = open_capture
        self._reconnect_delay = reconnect_delay
        self._clock = clock
        self._cond = threading.Condition()
        self._snapshot: Snapshot | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="rtsp-reader", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout)

    def latest(self) -> Snapshot | None:
        with self._cond:
            return self._snapshot

    def wait_for_new(self, last_seq: int, timeout: float = 1.0) -> Snapshot | None:
        """Block until a frame newer than `last_seq` arrives; None on timeout or stop."""
        with self._cond:
            self._cond.wait_for(
                lambda: self._stop.is_set()
                or (self._snapshot is not None and self._snapshot.seq > last_seq),
                timeout,
            )
            snap = self._snapshot
            return snap if snap is not None and snap.seq > last_seq else None

    def _publish(self, frame: Any) -> None:
        with self._cond:
            seq = self._snapshot.seq + 1 if self._snapshot else 1
            self._snapshot = Snapshot(frame, seq, self._clock())
            self._cond.notify_all()

    def _run(self) -> None:
        while not self._stop.is_set():
            capture = None
            try:
                capture = self._open_capture()
                if not capture.isOpened():
                    raise ConnectionError("cannot open the stream")
                log.info("Stream connected")
                while not self._stop.is_set():
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        log.warning("Stream lost")
                        break
                    self._publish(frame)
            except Exception as exc:
                log.warning("Capture error: %s", exc)
            finally:
                if capture is not None:
                    try:
                        capture.release()
                    except Exception:
                        log.debug("release() failed", exc_info=True)
            self._stop.wait(self._reconnect_delay)


def make_rtsp_capture(url: str, transport: str = "tcp", timeout_sec: float = 5.0):
    """Open an RTSP stream through OpenCV's FFmpeg backend with low-latency options."""
    import cv2

    # Read by the FFmpeg backend when a capture is opened, so it must be set before that.
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        f"rtsp_transport;{transport}|fflags;nobuffer|flags;low_delay"
    )
    timeout_ms = int(timeout_sec * 1000)
    capture = cv2.VideoCapture(
        url,
        cv2.CAP_FFMPEG,
        [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms, cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms],
    )
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture
