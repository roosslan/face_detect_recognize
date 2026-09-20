import threading

import numpy as np

from facerec.capture import LatestFrameReader
from facerec.faces import KnownFaces
from facerec.recognizer import UNKNOWN, FaceRecognizer, RecognitionWorker


def vec(value: float) -> np.ndarray:
    return np.full(128, value)


KNOWN = KnownFaces(["ann"], vec(0.0).reshape(1, -1))


class FakeEngine:
    def __init__(self, faces):
        self.faces = faces
        self.frames = []

    def detect(self, frame_bgr):
        self.frames.append(frame_bgr)
        return self.faces


class FakeReader:
    """Serves numbered frames and blocks like the real reader when nothing is new."""

    def __init__(self, frames):
        self._frames = list(frames)
        self._stop = threading.Event()

    def wait_for_new(self, last_seq, timeout=1.0):
        from facerec.capture import Snapshot

        if last_seq < len(self._frames):
            return Snapshot(self._frames[last_seq], last_seq + 1, 0.0)
        self._stop.wait(min(timeout, 0.05))
        return None


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_identify_names_known_and_unknown_faces():
    engine = FakeEngine([((1, 2, 3, 4), vec(0.01)), ((5, 6, 7, 8), vec(1.0))])
    result = FaceRecognizer(engine, KNOWN, tolerance=0.6).identify("frame")
    assert [d.name for d in result] == ["ann", UNKNOWN]
    assert result[0].box == (1, 2, 3, 4)
    assert result[0].distance < 0.6 < result[1].distance
    assert engine.frames == ["frame"]


def test_identify_with_no_faces_in_frame():
    assert FaceRecognizer(FakeEngine([]), KNOWN).identify("frame") == []


def test_everybody_is_unknown_when_nobody_is_registered():
    empty = KnownFaces([], np.empty((0, 128)))
    result = FaceRecognizer(FakeEngine([((0, 1, 1, 0), vec(0.0))]), empty).identify("f")
    assert [d.name for d in result] == [UNKNOWN]
    assert result[0].distance is None


def wait_for(predicate, timeout=2.0):
    done = threading.Event()
    for _ in range(int(timeout / 0.01)):
        if predicate():
            return True
        done.wait(0.01)
    return False


def test_worker_processes_frames_and_notifies_handler():
    engine = FakeEngine([((1, 2, 3, 4), vec(0.0))])
    seen = []
    worker = RecognitionWorker(
        FakeReader(["f1", "f2"]), FaceRecognizer(engine, KNOWN), on_detections=seen.append
    )
    worker.start()
    try:
        assert wait_for(lambda: len(seen) == 2)
    finally:
        worker.stop()
    assert engine.frames == ["f1", "f2"]
    assert [d.name for d in worker.latest_detections()] == ["ann"]


def test_worker_survives_engine_and_handler_errors():
    class Flaky(FakeEngine):
        def detect(self, frame_bgr):
            if frame_bgr == "bad":
                raise RuntimeError("dlib exploded")
            return super().detect(frame_bgr)

    def bad_handler(_):
        raise ValueError("handler exploded")

    engine = Flaky([((1, 2, 3, 4), vec(0.0))])
    worker = RecognitionWorker(
        FakeReader(["bad", "good"]), FaceRecognizer(engine, KNOWN), on_detections=bad_handler
    )
    worker.start()
    try:
        # the worker sleeps 1 s after an engine error, then still handles the next frame
        assert wait_for(lambda: engine.frames == ["good"], timeout=4.0)
    finally:
        worker.stop()


def test_stale_detections_are_hidden():
    clock = Clock()
    worker = RecognitionWorker(
        FakeReader([]), FaceRecognizer(FakeEngine([]), KNOWN), clock=clock
    )
    assert worker.latest_detections() == []  # nothing recognized yet

    worker._result = ([object()], clock())
    assert len(worker.latest_detections(max_age=1.0)) == 1
    clock.now += 1.5
    assert worker.latest_detections(max_age=1.0) == []


def test_worker_works_with_a_real_reader():
    class OneFrameCapture:
        def __init__(self):
            self._sent = False
            self._release = threading.Event()

        def isOpened(self):
            return True

        def read(self):
            if not self._sent:
                self._sent = True
                return True, "frame"
            self._release.wait(0.2)
            return False, None

        def release(self):
            self._release.set()

    reader = LatestFrameReader(OneFrameCapture)
    engine = FakeEngine([((1, 2, 3, 4), vec(0.0))])
    worker = RecognitionWorker(reader, FaceRecognizer(engine, KNOWN))
    reader.start()
    worker.start()
    try:
        assert wait_for(lambda: worker.latest_detections() != [])
    finally:
        worker.stop()
        reader.stop()
