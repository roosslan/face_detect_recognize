import pytest

from facerec import app


def test_defaults():
    args = app.build_parser().parse_args([])
    assert args.config == "config.toml"
    assert args.headless is False
    assert args.list is None


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
