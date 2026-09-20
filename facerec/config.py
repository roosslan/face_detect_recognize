"""Configuration loading and validation (TOML)."""

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


class ConfigError(Exception):
    """Raised when the configuration file is missing or invalid."""


@dataclass(frozen=True)
class CameraConfig:
    rtsp_url: str = ""
    transport: str = "tcp"  # "tcp" is reliable, "udp" has slightly lower latency
    reconnect_delay_sec: float = 3.0
    timeout_sec: float = 5.0  # open/read timeout, a hung camera triggers a reconnect


@dataclass(frozen=True)
class RedisConfig:
    host: str = "127.0.0.1"
    port: int = 6379
    db: int = 0
    password: str = ""
    stream_key: str = "face_events"
    stream_maxlen: int = 10_000


@dataclass(frozen=True)
class RecognitionConfig:
    faces_dir: str = "faces"
    tolerance: float = 0.6  # max face distance still counted as a match (lower = stricter)
    detect_scale: float = 0.5  # frames are downscaled by this factor before detection
    upsample: int = 1  # detector upsampling passes: +1 finds smaller faces, but is ~4x slower
    model: str = "hog"  # "hog" (CPU, fast) or "cnn" (more accurate, needs a GPU)
    min_interval_sec: float = 0.0  # throttle: minimum time between two recognition passes


@dataclass(frozen=True)
class EventsConfig:
    log_cooldown_sec: float = 10.0  # log the same person at most once per N seconds
    log_unknown: bool = False


@dataclass(frozen=True)
class SnapshotsConfig:
    enabled: bool = True
    directory: str = "capture"
    cooldown_sec: float = 10.0  # after a snapshot, wait at least this long for the next one
    settle_sec: float = 1.5  # after motion starts, wait this long for a face to be recognized
    motion_area: float = 0.005  # share of the frame that must change to count as motion
    motion_delta: float = 25.0  # change (0-255) of a colour channel that makes a pixel "changed"
    annotate: bool = True  # draw boxes and names on the snapshot


@dataclass(frozen=True)
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    events: EventsConfig = field(default_factory=EventsConfig)
    snapshots: SnapshotsConfig = field(default_factory=SnapshotsConfig)


def redact_url(url: str) -> str:
    """Hide credentials in a URL so it can be logged safely."""
    parts = urlsplit(url)
    if "@" not in parts.netloc:
        return url
    host = parts.netloc.rsplit("@", 1)[1]
    return urlunsplit(parts._replace(netloc=f"***@{host}"))


def load_config(path: str | Path) -> Config:
    path = Path(path)
    try:
        with path.open("rb") as fh:
            raw = tomllib.load(fh)
    except FileNotFoundError:
        raise ConfigError(
            f"Config file '{path}' not found. Copy config.example.toml to {path} and edit it."
        ) from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in '{path}': {exc}") from exc
    return parse_config(raw)


def parse_config(raw: dict[str, Any]) -> Config:
    sections = {f.name: f for f in dataclasses.fields(Config)}
    unknown = set(raw) - set(sections)
    if unknown:
        raise ConfigError(f"Unknown config section(s): {', '.join(sorted(unknown))}")

    built = {
        name: _build_section(sections[name].default_factory, name, raw.get(name, {}))
        for name in sections
    }
    config = Config(**built)
    _validate(config)
    return config


def _build_section(cls: type, name: str, data: Any) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"[{name}] must be a table")
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ConfigError(f"Unknown key(s) in [{name}]: {', '.join(sorted(unknown))}")
    for key, value in data.items():
        expected = type(getattr(cls(), key))
        # ints are acceptable where floats are expected (e.g. `tolerance = 1`)
        if expected is float and isinstance(value, int) and not isinstance(value, bool):
            continue
        if not isinstance(value, expected) or (expected is int and isinstance(value, bool)):
            raise ConfigError(
                f"[{name}] {key} must be {expected.__name__}, got {type(value).__name__}"
            )
    return cls(**data)


def _validate(config: Config) -> None:
    cam, rd, rec, ev = config.camera, config.redis, config.recognition, config.events
    snap = config.snapshots
    if not cam.rtsp_url:
        raise ConfigError("[camera] rtsp_url is required")
    _require(cam.transport in ("tcp", "udp"), "[camera] transport must be 'tcp' or 'udp'")
    _require(cam.reconnect_delay_sec >= 0, "[camera] reconnect_delay_sec must be >= 0")
    _require(cam.timeout_sec > 0, "[camera] timeout_sec must be > 0")
    _require(1 <= rd.port <= 65535, "[redis] port must be in 1..65535")
    _require(rd.db >= 0, "[redis] db must be >= 0")
    _require(bool(rd.stream_key), "[redis] stream_key must not be empty")
    _require(rd.stream_maxlen > 0, "[redis] stream_maxlen must be > 0")
    _require(bool(rec.faces_dir), "[recognition] faces_dir must not be empty")
    _require(0 < rec.tolerance <= 1, "[recognition] tolerance must be in (0, 1]")
    _require(0 < rec.detect_scale <= 1, "[recognition] detect_scale must be in (0, 1]")
    _require(0 <= rec.upsample <= 3, "[recognition] upsample must be in 0..3")
    _require(rec.model in ("hog", "cnn"), "[recognition] model must be 'hog' or 'cnn'")
    _require(rec.min_interval_sec >= 0, "[recognition] min_interval_sec must be >= 0")
    _require(ev.log_cooldown_sec >= 0, "[events] log_cooldown_sec must be >= 0")
    _require(bool(snap.directory), "[snapshots] directory must not be empty")
    _require(snap.cooldown_sec >= 0, "[snapshots] cooldown_sec must be >= 0")
    _require(snap.settle_sec >= 0, "[snapshots] settle_sec must be >= 0")
    _require(0 < snap.motion_area < 1, "[snapshots] motion_area must be in (0, 1)")
    _require(0 < snap.motion_delta <= 255, "[snapshots] motion_delta must be in (0, 255]")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)
