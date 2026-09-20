import threading

from facerec.capture import LatestFrameReader


class ScriptedCapture:
    """Yields the given frames, then either reports a lost stream or blocks until released."""

    def __init__(self, frames, opened=True, then_fail=False):
        self._frames = list(frames)
        self._opened = opened
        self._then_fail = then_fail
        self.released = threading.Event()

    def isOpened(self):
        return self._opened

    def read(self):
        if self._frames:
            return True, self._frames.pop(0)
        if self._then_fail:
            return False, None
        self.released.wait(0.2)  # emulate a live stream that has no new data
        return False, None

    def release(self):
        self.released.set()


def wait_until(reader, seq):
    """Wait for the frame with the given sequence number to be published."""
    snap = reader.latest()
    while snap is None or snap.seq < seq:
        snap = reader.wait_for_new(snap.seq if snap else 0, timeout=2)
        assert snap is not None, f"frame {seq} never arrived"
    return snap


def test_latest_returns_none_before_any_frame():
    reader = LatestFrameReader(lambda: ScriptedCapture([]))
    assert reader.latest() is None


def test_reader_exposes_the_newest_frame_and_drops_older_ones():
    capture = ScriptedCapture(["f1", "f2", "f3"])
    reader = LatestFrameReader(lambda: capture)
    reader.start()
    try:
        snap = wait_until(reader, 3)
        assert (snap.frame, snap.seq) == ("f3", 3)
        assert reader.latest() == snap
    finally:
        reader.stop()


def test_wait_for_new_returns_none_when_nothing_newer_arrives():
    capture = ScriptedCapture(["f1"])
    reader = LatestFrameReader(lambda: capture)
    reader.start()
    try:
        snap = wait_until(reader, 1)
        assert reader.wait_for_new(snap.seq, timeout=0.05) is None
    finally:
        reader.stop()


def test_wait_for_new_returns_immediately_if_a_newer_frame_exists():
    capture = ScriptedCapture(["f1", "f2"])
    reader = LatestFrameReader(lambda: capture)
    reader.start()
    try:
        wait_until(reader, 2)
        snap = reader.wait_for_new(1, timeout=0.05)
        assert snap is not None and snap.frame == "f2"
    finally:
        reader.stop()


def test_reader_reconnects_after_the_stream_is_lost():
    captures = [
        ScriptedCapture(["a1"], then_fail=True),
        ScriptedCapture(["b1", "b2"]),
    ]
    opened = []

    def factory():
        capture = captures[len(opened)]
        opened.append(capture)
        return capture

    reader = LatestFrameReader(factory, reconnect_delay=0)
    reader.start()
    try:
        snap = wait_until(reader, 3)
        assert snap.frame == "b2"
        assert len(opened) == 2
        assert captures[0].released.is_set()  # the broken capture was released
    finally:
        reader.stop()


def test_reader_retries_when_the_stream_cannot_be_opened():
    captures = [ScriptedCapture([], opened=False), ScriptedCapture(["ok"])]
    opened = []

    def factory():
        capture = captures[len(opened)]
        opened.append(capture)
        return capture

    reader = LatestFrameReader(factory, reconnect_delay=0)
    reader.start()
    try:
        assert wait_until(reader, 1).frame == "ok"
    finally:
        reader.stop()


def test_reader_survives_an_exception_from_the_factory():
    calls = []

    def factory():
        calls.append(1)
        if len(calls) == 1:
            raise OSError("boom")
        return ScriptedCapture(["ok"])

    reader = LatestFrameReader(factory, reconnect_delay=0)
    reader.start()
    try:
        assert wait_until(reader, 1).frame == "ok"
    finally:
        reader.stop()


def test_stop_ends_the_thread():
    capture = ScriptedCapture([])
    reader = LatestFrameReader(lambda: capture)
    reader.start()
    reader.stop()
    assert not reader._thread.is_alive()
