import pytest

from facerec.config import ConfigError, load_config, parse_config, redact_url


def test_defaults_are_applied():
    cfg = parse_config({"camera": {"rtsp_url": "rtsp://cam/1"}})
    assert cfg.camera.transport == "tcp"
    assert cfg.redis.port == 6379
    assert cfg.recognition.tolerance == 0.6
    assert cfg.events.log_unknown is False


def test_snapshot_defaults():
    snap = parse_config({"camera": {"rtsp_url": "x"}}).snapshots
    assert snap.enabled is True
    assert snap.directory == "capture"
    assert snap.annotate is True


def test_snapshots_can_be_disabled_and_redirected():
    cfg = parse_config(
        {"camera": {"rtsp_url": "x"}, "snapshots": {"enabled": False, "directory": "shots"}}
    )
    assert cfg.snapshots.enabled is False
    assert cfg.snapshots.directory == "shots"


def test_values_are_overridden():
    cfg = parse_config(
        {
            "camera": {"rtsp_url": "rtsp://cam/1", "transport": "udp"},
            "redis": {"host": "redis.local", "port": 6380},
            "events": {"log_unknown": True},
        }
    )
    assert cfg.camera.transport == "udp"
    assert (cfg.redis.host, cfg.redis.port) == ("redis.local", 6380)
    assert cfg.events.log_unknown is True


def test_int_is_accepted_for_float_field():
    cfg = parse_config({"camera": {"rtsp_url": "x"}, "recognition": {"tolerance": 1}})
    assert cfg.recognition.tolerance == 1


def test_rtsp_url_is_required():
    with pytest.raises(ConfigError, match="rtsp_url"):
        parse_config({})


@pytest.mark.parametrize(
    "section, key, value, message",
    [
        ("camera", "transport", "http", "transport"),
        ("camera", "timeout_sec", 0, "timeout_sec"),
        ("redis", "port", 70000, "port"),
        ("recognition", "tolerance", 0, "tolerance"),
        ("recognition", "detect_scale", 1.5, "detect_scale"),
        ("recognition", "upsample", 4, "upsample"),
        ("recognition", "upsample", -1, "upsample"),
        ("recognition", "model", "svm", "model"),
        ("events", "log_cooldown_sec", -1, "log_cooldown_sec"),
        ("snapshots", "directory", "", "directory"),
        ("snapshots", "cooldown_sec", -1, "cooldown_sec"),
        ("snapshots", "settle_sec", -0.5, "settle_sec"),
        ("snapshots", "motion_area", 0, "motion_area"),
        ("snapshots", "motion_area", 1, "motion_area"),
        ("snapshots", "motion_delta", 300, "motion_delta"),
    ],
)
def test_invalid_values_are_rejected(section, key, value, message):
    raw = {"camera": {"rtsp_url": "x"}}
    raw.setdefault(section, {})[key] = value
    with pytest.raises(ConfigError, match=message):
        parse_config(raw)


def test_unknown_key_is_rejected():
    with pytest.raises(ConfigError, match="tolerence"):
        parse_config({"camera": {"rtsp_url": "x"}, "recognition": {"tolerence": 0.5}})


def test_unknown_section_is_rejected():
    with pytest.raises(ConfigError, match="camra"):
        parse_config({"camra": {}})


def test_wrong_type_is_rejected():
    with pytest.raises(ConfigError, match="port must be int"):
        parse_config({"camera": {"rtsp_url": "x"}, "redis": {"port": "6379"}})


def test_bool_is_not_accepted_as_int():
    with pytest.raises(ConfigError, match="port must be int"):
        parse_config({"camera": {"rtsp_url": "x"}, "redis": {"port": True}})


def test_load_config_reads_toml_file(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[camera]\nrtsp_url = "rtsp://cam/2"\n', encoding="utf-8")
    assert load_config(path).camera.rtsp_url == "rtsp://cam/2"


def test_load_config_missing_file_points_to_example(tmp_path):
    with pytest.raises(ConfigError, match=r"config\.example\.toml"):
        load_config(tmp_path / "nope.toml")


def test_load_config_invalid_toml(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("[camera\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Invalid TOML"):
        load_config(path)


def test_redact_url_hides_credentials():
    assert redact_url("rtsp://admin:secret@10.0.0.5:554/1") == "rtsp://***@10.0.0.5:554/1"


def test_redact_url_keeps_plain_url():
    assert redact_url("rtsp://10.0.0.5:554/1") == "rtsp://10.0.0.5:554/1"
