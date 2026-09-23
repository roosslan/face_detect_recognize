"""Detection events in a Redis Stream."""

import logging
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

from facerec.config import RedisConfig
from facerec.recognizer import UNKNOWN, Detection
from facerec.snapshots import SnapshotWriter

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
    """Writes "person seen" events, at most one per person per `cooldown` seconds.

    When `snapshots` is given, every event that actually gets written is backed by a picture:
    the frame that was being recognized when it fired is saved the same way a motion snapshot
    is (same file naming, same directory), so a Redis entry is never left without evidence of
    what triggered it."""

    def __init__(
        self,
        client: Any,
        stream_key: str,
        maxlen: int,
        cooldown: float,
        log_unknown: bool = False,
        snapshots: SnapshotWriter | None = None,
        annotate: Callable[[Any, list[Detection]], Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = datetime.now,
    ):
        self._client = client
        self._stream_key = stream_key
        self._maxlen = maxlen
        self._cooldown = cooldown
        self._log_unknown = log_unknown
        self._snapshots = snapshots
        self._annotate = annotate
        self._clock = clock
        self._now = now
        self._last_logged: dict[str, float] = {}

    def handle(self, detections: list[Detection], frame: Any = None) -> None:
        when = self._now()
        logged = False
        for name in dict.fromkeys(d.name for d in detections):  # unique, order kept
            if name == UNKNOWN and not self._log_unknown:
                continue
            mark = self._clock()
            last = self._last_logged.get(name)
            if last is not None and mark - last < self._cooldown:
                continue
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
            logged = True
            log.info("Event: %s @ %s (id %s)", name, when.strftime("%H:%M:%S"), entry_id)
        # One snapshot per call, not per name: several people logged from the same frame would
        # otherwise each get their own, near-identical picture of the same moment.
        if logged and self._snapshots is not None and frame is not None:
            self._save_snapshot(frame, detections, when)

    def _save_snapshot(self, frame: Any, detections: list[Detection], when: datetime) -> None:
        # The Redis write above already succeeded; a camera without a picture is still better
        # than no event at all, so a failure here is only logged, never raised.
        try:
            image = self._annotate(frame, detections) if self._annotate else frame
            names = sorted({d.name for d in detections if d.name != UNKNOWN})
            path = self._snapshots.save(image, names, when)
        except Exception:
            log.exception("Could not save the snapshot for this event")
        else:
            log.info("Snapshot: %s", path)
