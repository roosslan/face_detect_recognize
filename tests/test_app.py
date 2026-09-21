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
