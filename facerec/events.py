"""Detection events in a Redis Stream."""

import logging
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

from facerec.config import RedisConfig
from facerec.recognizer import UNKNOWN, Detection

log = logging.getLogger(__name__)


def connect_redis(cfg: RedisConfig):
    """Create a client and ping the server; raises if it is unreachable."""
    import redis

    client = redis.Redis(
        host=cfg.host,
        port=cfg.port,
        db=cfg.db,
        password=cfg.password or None,
        decode_responses=True,
        socket_connect_timeout=3,
        socket_timeout=3,
    )
    client.ping()
    return client


def read_events(client: Any, stream_key: str, count: int = 20) -> list[tuple[str, dict]]:
    """Latest `count` events, newest first."""
    return client.xrevrange(stream_key, count=count)


class EventLogger:
    """Writes "person seen" events, at most one per person per `cooldown` seconds."""

    def __init__(
        self,
        client: Any,
        stream_key: str,
        maxlen: int,
        cooldown: float,
        log_unknown: bool = False,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = datetime.now,
    ):
        self._client = client
        self._stream_key = stream_key
        self._maxlen = maxlen
        self._cooldown = cooldown
        self._log_unknown = log_unknown
        self._clock = clock
        self._now = now
        self._last_logged: dict[str, float] = {}

    def handle(self, detections: list[Detection]) -> None:
        for name in dict.fromkeys(d.name for d in detections):  # unique, order kept
            if name == UNKNOWN and not self._log_unknown:
                continue
            mark = self._clock()
            last = self._last_logged.get(name)
            if last is not None and mark - last < self._cooldown:
                continue
            when = self._now()
            try:
                entry_id = self._client.xadd(
                    self._stream_key,
                    {"time": when.strftime("%Y-%m-%d %H:%M:%S"), "name": name},
                    maxlen=self._maxlen,
                    approximate=True,
                )
            except Exception as exc:
                # cooldown is not started, so the write is retried on the next frame
                log.error("Redis write failed: %s", exc)
                continue
            self._last_logged[name] = mark
            log.info("Event: %s @ %s (id %s)", name, when.strftime("%H:%M:%S"), entry_id)
