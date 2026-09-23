import os
import sys

import pytest

from facerec import app


def test_defaults():
    args = app.build_parser().parse_args([])
    assert args.config is None
    assert args.headless is False
    assert args.list is None


def test_self_test_flag():
    assert app.build_parser().parse_args(["--self-test"]).self_test is True
    assert app.build_parser().parse_args([]).self_test is False


def test_list_without_count_defaults_to_twenty():
    assert app.build_parser().parse_args(["--list"]).list == 20


def test_list_with_count():
    assert app.build_parser().parse_args(["--list", "5"]).list == 5


def test_flags():
    args = app.build_parser().parse_args(["--headless", "--config", "other.toml"])
    assert args.headless is True
    assert args.config == "other.toml"


def test_missing_config_exits_with_code_2(tmp_path, caplog):
    assert app.main(["--config", str(tmp_path / "missing.toml")]) == 2
    assert "config.example.toml" in caplog.text


def test_invalid_option_is_rejected():
    with pytest.raises(SystemExit):
        app.build_parser().parse_args(["--list", "many"])


def test_script_uses_config_from_the_current_folder(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    assert app.main([]) == 2
    assert str(tmp_path / "config.toml") in caplog.text


def test_frozen_exe_works_in_its_own_folder(tmp_path, monkeypatch, caplog):
    exe_dir, elsewhere = tmp_path / "app", tmp_path / "elsewhere"
    exe_dir.mkdir()
    elsewhere.mkdir()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe_dir / "facerec.exe"))
    monkeypatch.chdir(elsewhere)  # also restores the folder after the test

    assert app.frozen_app_dir() == exe_dir
    assert app.main([]) == 2
    assert str(exe_dir / "config.toml") in caplog.text  # not looked for in "elsewhere"
    assert os.getcwd() == str(exe_dir)  # faces/ and capture/ resolve next to the exe


def test_frozen_exe_takes_an_explicit_config_relative_to_where_it_was_typed(
    tmp_path, monkeypatch, caplog
):
    exe_dir, elsewhere = tmp_path / "app", tmp_path / "elsewhere"
    exe_dir.mkdir()
    elsewhere.mkdir()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe_dir / "facerec.exe"))
    monkeypatch.chdir(elsewhere)

    assert app.main(["--config", "mine.toml"]) == 2
    assert str(elsewhere / "mine.toml") in caplog.text


def test_frozen_app_dir_is_none_for_a_script():
    assert app.frozen_app_dir() is None


def test_self_test_passes_when_nothing_raises(monkeypatch):
    monkeypatch.setattr(app, "run_self_test", lambda: None)
    assert app.main(["--self-test"]) == 0


def test_self_test_reports_a_failure_with_exit_code_1(monkeypatch, caplog):
    def broken():
        raise ImportError("no dlib")

    monkeypatch.setattr(app, "run_self_test", broken)
    assert app.main(["--self-test"]) == 1
    assert "Self-test failed" in caplog.text


def test_self_test_does_not_need_a_config(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "run_self_test", lambda: None)
    monkeypatch.chdir(tmp_path)  # no config.toml here
    assert app.main(["--self-test"]) == 0


# --- run_viewer / the preview window ---


class FakeCv2:
    """Stands in for cv2 for run_viewer: no real display needed. WND_PROP_VISIBLE and
    FONT_HERSHEY_DUPLEX are arbitrary distinct sentinels, never interpreted, just passed
    through the way the real constants would be."""

    error = type("error", (Exception,), {})
    WND_PROP_VISIBLE = 0
    FONT_HERSHEY_DUPLEX = 1

    def __init__(self, closed_after=None, property_raises_after=None):
        self.closed_after = closed_after  # frame count after which the window "closes" (< 1)
        self.property_raises_after = property_raises_after  # ...after which it raises instead
        self.frames_shown = 0
        self.destroyed = False

    def imshow(self, _window, _canvas):
        self.frames_shown += 1

    def waitKey(self, _ms):
        return -1  # nobody pressed q

    def getWindowProperty(self, _window, _prop):
        raises_after = self.property_raises_after
        if raises_after is not None and self.frames_shown > raises_after:
            raise self.error("NULL guiReceiver (please create a window)")
        if self.closed_after is not None and self.frames_shown > self.closed_after:
            return -1.0
        return 1.0

    def putText(self, *args, **kwargs):
        pass

    def destroyAllWindows(self):
        self.destroyed = True


class FakeReader:
    def __init__(self, frames):
        self._frames = list(frames)
        self._sent = 0

    def wait_for_new(self, _last_seq, timeout=0.1):
        from facerec.capture import Snapshot

        if self._sent < len(self._frames):
            self._sent += 1
            return Snapshot(self._frames[self._sent - 1], self._sent, 0.0)
        return None

    def latest(self):
        return None


class FakeWorker:
    def latest_detections(self):
        return []


def frame():
    import numpy as np

    return np.zeros((10, 10, 3), dtype=np.uint8)


def test_viewer_stops_normally_when_the_window_reports_closed(monkeypatch):
    fake_cv2 = FakeCv2(closed_after=2)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    app.run_viewer(FakeReader([frame(), frame(), frame(), frame()]), FakeWorker())
    assert fake_cv2.frames_shown == 3  # stopped as soon as the property read reported closed
    assert fake_cv2.destroyed is True


def test_viewer_treats_a_getWindowProperty_crash_as_closed_not_an_error(monkeypatch):
    """Regression test: on some OpenCV/Qt builds, closing the window with the X button tears
    it down immediately, and the next getWindowProperty() raises cv2.error("NULL guiReceiver")
    instead of returning < 1. That must stop the loop cleanly, not crash the program."""
    fake_cv2 = FakeCv2(property_raises_after=2)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    app.run_viewer(FakeReader([frame(), frame(), frame(), frame()]), FakeWorker())  # must not raise
    assert fake_cv2.frames_shown == 3
    assert fake_cv2.destroyed is True
