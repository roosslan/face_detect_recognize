from datetime import datetime

from facerec.events import EventLogger, read_events
from facerec.recognizer import UNKNOWN, Detection

BOX = (0, 10, 10, 0)


def det(name: str) -> Detection:
    return Detection(name, BOX)


class FakeRedis:
    def __init__(self):
        self.entries: list[tuple[str, str, dict]] = []
        self.fail = False
        self.kwargs: dict = {}

    def xadd(self, key, fields, **kwargs):
        if self.fail:
            raise ConnectionError("redis is down")
        self.kwargs = kwargs
        self.entries.append((key, f"{len(self.entries) + 1}-0", fields))
        return self.entries[-1][1]

    def xrevrange(self, key, count):
        rows = [(entry_id, fields) for k, entry_id, fields in self.entries if k == key]
        return rows[::-1][:count]


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def make_logger(client, clock, **overrides):
    params = dict(
        stream_key="face_events",
        maxlen=500,
        cooldown=10.0,
        clock=clock,
        now=lambda: datetime(2026, 9, 12, 13, 21, 0),
    )
    params.update(overrides)
    return EventLogger(client, **params)


def test_event_has_time_and_name_fields():
    client = FakeRedis()
    make_logger(client, Clock()).handle([det("rasa")])
    key, _, fields = client.entries[0]
    assert key == "face_events"
    assert fields == {"time": "2026-09-12 13:21:00", "name": "rasa"}


def test_stream_is_capped():
    client = FakeRedis()
    make_logger(client, Clock(), maxlen=500).handle([det("rasa")])
    assert client.kwargs == {"maxlen": 500, "approximate": True}


def test_same_person_is_logged_once_per_cooldown():
    client, clock = FakeRedis(), Clock()
    logger = make_logger(client, clock)
    logger.handle([det("rasa")])
    clock.now += 9.9
    logger.handle([det("rasa")])
    assert len(client.entries) == 1

    clock.now += 0.2  # 10.1 s since the first event
    logger.handle([det("rasa")])
    assert len(client.entries) == 2


def test_people_have_independent_cooldowns():
    client, clock = FakeRedis(), Clock()
    logger = make_logger(client, clock)
    logger.handle([det("rasa")])
    clock.now += 1
    logger.handle([det("tima")])
    assert [e[2]["name"] for e in client.entries] == ["rasa", "tima"]


def test_duplicate_names_in_one_frame_are_logged_once():
    client = FakeRedis()
    make_logger(client, Clock()).handle([det("rasa"), det("rasa")])
    assert len(client.entries) == 1


def test_unknown_faces_are_skipped_by_default():
    client = FakeRedis()
    make_logger(client, Clock()).handle([det(UNKNOWN)])
    assert client.entries == []


def test_unknown_faces_are_logged_when_enabled():
    client = FakeRedis()
    make_logger(client, Clock(), log_unknown=True).handle([det(UNKNOWN)])
    assert client.entries[0][2]["name"] == UNKNOWN


def test_failed_write_does_not_raise_and_is_retried_immediately():
    client, clock = FakeRedis(), Clock()
    logger = make_logger(client, clock)
    client.fail = True
    logger.handle([det("rasa")])
    assert client.entries == []

    client.fail = False  # no time has passed, the cooldown must not have started
    logger.handle([det("rasa")])
    assert len(client.entries) == 1


def test_read_events_returns_newest_first():
    client, clock = FakeRedis(), Clock()
    logger = make_logger(client, clock)
    for name in ("a", "b", "c"):
        logger.handle([det(name)])
    rows = read_events(client, "face_events", count=2)
    assert [fields["name"] for _, fields in rows] == ["c", "b"]
