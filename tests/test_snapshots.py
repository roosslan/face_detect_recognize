from datetime import datetime

import pytest

from facerec.recognizer import UNKNOWN, Detection
from facerec.snapshots import MotionSnapshotter, MotionWorker, SnapshotWriter, snapshot_stem

BOX = (0, 10, 10, 0)
WHEN = datetime(2026, 9, 20, 14, 3, 27)


def det(name):
    return Detection(name, BOX)


class FakeDetector:
    """Motion on/off is set by the test."""

    def __init__(self):
        self.moving = False

    def update(self, frame):
        return self.moving


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class Env:
    def __init__(self, tmp_path, **kwargs):
        self.detector = FakeDetector()
        self.clock = Clock()
        self.detections: list[Detection] = []
        self.encoded: list = []
        writer = SnapshotWriter(tmp_path / "capture", encode=self._encode)
        self.snapshotter = MotionSnapshotter(
            self.detector,
            writer,
            lambda: self.detections,
            clock=self.clock,
            now=lambda: WHEN,
            **kwargs,
        )
        self.dir = tmp_path / "capture"

    def _encode(self, frame):
        self.encoded.append(frame)
        return b"jpeg:" + str(frame).encode()

    def step(self, dt=0.1, frame="frame"):
        self.clock.now += dt
        return self.snapshotter.process(frame)

    def files(self):
        return sorted(p.name for p in self.dir.iterdir()) if self.dir.exists() else []


# --- file names ---


def test_stem_without_person():
    assert snapshot_stem(WHEN, []) == "20-09-2026-14-03"


def test_stem_with_person():
    assert snapshot_stem(WHEN, ["rasa"]) == "20-09-2026-14-03-rasa"


def test_stem_with_several_people_is_sorted_and_joined():
    assert snapshot_stem(WHEN, ["tima", "rasa"]) == "20-09-2026-14-03-rasa+tima"


def test_stem_replaces_characters_that_are_invalid_in_file_names():
    assert snapshot_stem(WHEN, ['a/b:c?']) == "20-09-2026-14-03-a_b_c_"


def test_stem_keeps_non_latin_names():
    assert snapshot_stem(WHEN, ["Расул"]) == "20-09-2026-14-03-Расул"


# --- writer ---


def test_writer_creates_the_directory_and_writes_the_encoded_frame(tmp_path):
    writer = SnapshotWriter(tmp_path / "a" / "b", encode=lambda f: b"data")
    path = writer.save("frame", ["rasa"], WHEN)
    assert path == tmp_path / "a" / "b" / "20-09-2026-14-03-rasa.jpg"
    assert path.read_bytes() == b"data"


def test_writer_never_overwrites_within_the_same_minute(tmp_path):
    writer = SnapshotWriter(tmp_path, encode=lambda f: f)
    first = writer.save(b"1", [], WHEN)
    second = writer.save(b"2", [], WHEN)
    third = writer.save(b"3", [], WHEN)
    assert [p.name for p in (first, second, third)] == [
        "20-09-2026-14-03.jpg",
        "20-09-2026-14-03 (2).jpg",
        "20-09-2026-14-03 (3).jpg",
    ]
    assert first.read_bytes() == b"1"


def test_encoding_failure_leaves_no_file_behind(tmp_path):
    def broken(_):
        raise ValueError("cannot encode")

    writer = SnapshotWriter(tmp_path / "capture", encode=broken)
    with pytest.raises(ValueError):
        writer.save("frame", [], WHEN)
    assert not (tmp_path / "capture").exists()


# --- when a snapshot is taken ---


def test_nothing_happens_without_motion(tmp_path):
    env = Env(tmp_path)
    assert all(env.step() is None for _ in range(50))
    assert env.files() == []


def test_recognized_person_is_named_in_the_file(tmp_path):
    env = Env(tmp_path, settle=1.5)
    env.detector.moving = True
    assert env.step() is None  # motion started, waiting for a face
    env.detections = [det("rasa")]
    path = env.step()
    assert path.name == "20-09-2026-14-03-rasa.jpg"


def test_snapshot_without_a_name_is_taken_when_the_wait_is_over(tmp_path):
    env = Env(tmp_path, settle=1.5)
    env.detector.moving = True
    for _ in range(14):
        assert env.step(0.1) is None  # 1.4 s: still waiting
    path = env.step(0.2)  # 1.6 s
    assert path.name == "20-09-2026-14-03.jpg"


def test_unknown_face_does_not_count_as_a_name(tmp_path):
    env = Env(tmp_path, settle=1.0)
    env.detector.moving = True
    env.detections = [det(UNKNOWN)]
    assert env.step(0.1) is None  # waits: maybe the face gets recognized
    path = env.step(1.0)
    assert path.name == "20-09-2026-14-03.jpg"


def test_zero_settle_takes_the_snapshot_right_away(tmp_path):
    env = Env(tmp_path, settle=0.0)
    env.detector.moving = True
    assert env.step().name == "20-09-2026-14-03.jpg"


def test_already_recognized_person_is_named_immediately(tmp_path):
    env = Env(tmp_path, settle=5.0)
    env.detections = [det("tima"), det("rasa")]
    env.detector.moving = True
    assert env.step().name == "20-09-2026-14-03-rasa+tima.jpg"


def test_cooldown_blocks_the_next_snapshot(tmp_path):
    env = Env(tmp_path, settle=0.0, cooldown=10.0)
    env.detector.moving = True
    assert env.step() is not None
    assert all(env.step(1.0) is None for _ in range(9))  # 9 s after the snapshot
    assert env.step(1.5) is not None  # 10.5 s
    assert len(env.files()) == 2  # same minute: the second one got " (2)"


def test_motion_that_ended_during_the_wait_still_gives_a_snapshot(tmp_path):
    env = Env(tmp_path, settle=1.0)
    env.detector.moving = True
    assert env.step() is None
    env.detector.moving = False  # the person left quickly
    path = env.step(1.5)
    assert path is not None


def test_annotate_is_applied_to_the_saved_image(tmp_path):
    env = Env(tmp_path, settle=0.0, annotate=lambda frame, dets: f"{frame}+boxes{len(dets)}")
    env.detections = [det("rasa")]
    env.detector.moving = True
    env.step(frame="F")
    assert env.encoded == ["F+boxes1"]


def test_failed_write_is_not_retried_on_every_frame(tmp_path):
    env = Env(tmp_path, settle=0.0, cooldown=10.0)
    calls = []

    def broken(frame):
        calls.append(1)
        raise OSError("disk full")

    env.snapshotter._writer._encode = broken
    env.detector.moving = True
    with pytest.raises(OSError):
        env.step()
    assert env.step() is None and env.step() is None
    assert len(calls) == 1


# --- worker ---


def test_worker_feeds_frames_to_the_snapshotter():
    from facerec.capture import LatestFrameReader

    class OneFrame:
        def __init__(self):
            import threading

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

    class Recorder:
        def __init__(self):
            self.frames = []

        def process(self, frame):
            self.frames.append(frame)

    import threading

    recorder = Recorder()
    reader = LatestFrameReader(OneFrame)
    worker = MotionWorker(reader, recorder)
    reader.start()
    worker.start()
    try:
        done = threading.Event()
        for _ in range(200):
            if recorder.frames:
                break
            done.wait(0.01)
        assert recorder.frames == ["frame"]
    finally:
        worker.stop()
        reader.stop()
