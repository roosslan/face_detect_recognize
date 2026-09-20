import json
import os

import numpy as np

from facerec.faces import CACHE_FILENAME, KnownFaces, load_known_faces


def vec(value: float) -> np.ndarray:
    return np.full(128, value)


class FakeEncoder:
    """Stands in for dlib: the 'photo' content decides the encoding."""

    def __init__(self):
        self.calls: list[str] = []

    def __call__(self, path):
        self.calls.append(path.name)
        text = path.read_text()
        return None if text == "noface" else vec(float(text))


def make_photo(folder, name, content):
    path = folder / name
    path.write_text(content)
    return path


# --- KnownFaces.match ---


def test_match_returns_closest_name_within_tolerance():
    known = KnownFaces(["ann", "bob"], np.vstack([vec(0.0), vec(1.0)]))
    name, distance = known.match(vec(0.01), tolerance=0.6)
    assert name == "ann"
    assert distance is not None and distance < 0.6


def test_match_picks_the_closest_of_several_candidates():
    known = KnownFaces(["ann", "bob"], np.vstack([vec(0.0), vec(0.04)]))
    assert known.match(vec(0.05), tolerance=1.0)[0] == "bob"


def test_match_beyond_tolerance_is_unknown_but_reports_distance():
    known = KnownFaces(["ann"], vec(0.0).reshape(1, -1))
    name, distance = known.match(vec(1.0), tolerance=0.6)
    assert name is None
    assert distance is not None and distance > 0.6


def test_match_with_no_known_faces():
    known = KnownFaces([], np.empty((0, 128)))
    assert known.match(vec(0.0), tolerance=0.6) == (None, None)


# --- load_known_faces ---


def test_missing_folder_gives_empty_result(tmp_path):
    known = load_known_faces(tmp_path / "absent", FakeEncoder())
    assert len(known) == 0


def test_names_come_from_file_stems_and_non_images_are_ignored(tmp_path):
    make_photo(tmp_path, "ann.jpg", "0.1")
    make_photo(tmp_path, "Bob.PNG", "0.2")
    make_photo(tmp_path, "notes.txt", "0.3")
    known = load_known_faces(tmp_path, FakeEncoder())
    assert known.names == ["Bob", "ann"]  # sorted by file name, case-sensitive
    assert known.encodings.shape == (2, 128)


def test_photo_without_a_face_is_skipped(tmp_path):
    make_photo(tmp_path, "ann.jpg", "0.1")
    make_photo(tmp_path, "empty.jpg", "noface")
    assert load_known_faces(tmp_path, FakeEncoder()).names == ["ann"]


def test_unreadable_photo_is_skipped(tmp_path):
    make_photo(tmp_path, "ann.jpg", "0.1")
    make_photo(tmp_path, "broken.jpg", "not a number")  # FakeEncoder raises ValueError
    assert load_known_faces(tmp_path, FakeEncoder()).names == ["ann"]


def test_second_load_uses_the_cache(tmp_path):
    make_photo(tmp_path, "ann.jpg", "0.1")
    first = FakeEncoder()
    load_known_faces(tmp_path, first)
    assert first.calls == ["ann.jpg"]

    second = FakeEncoder()
    known = load_known_faces(tmp_path, second)
    assert second.calls == []
    assert known.names == ["ann"]
    assert np.allclose(known.encodings[0], vec(0.1))


def test_only_changed_photos_are_re_encoded(tmp_path):
    ann = make_photo(tmp_path, "ann.jpg", "0.1")
    make_photo(tmp_path, "bob.jpg", "0.2")
    load_known_faces(tmp_path, FakeEncoder())

    ann.write_text("0.15")  # new size, so the cache entry is stale
    os.utime(ann, ns=(ann.stat().st_atime_ns, ann.stat().st_mtime_ns + 1_000_000_000))
    encoder = FakeEncoder()
    known = load_known_faces(tmp_path, encoder)
    assert encoder.calls == ["ann.jpg"]
    assert np.allclose(known.encodings[known.names.index("ann")], vec(0.15))


def test_removed_photo_is_dropped_from_cache(tmp_path):
    make_photo(tmp_path, "ann.jpg", "0.1")
    bob = make_photo(tmp_path, "bob.jpg", "0.2")
    load_known_faces(tmp_path, FakeEncoder())
    bob.unlink()

    assert load_known_faces(tmp_path, FakeEncoder()).names == ["ann"]
    cached = json.loads((tmp_path / CACHE_FILENAME).read_text())["entries"]
    assert list(cached) == ["ann.jpg"]


def test_corrupted_cache_is_ignored(tmp_path):
    make_photo(tmp_path, "ann.jpg", "0.1")
    (tmp_path / CACHE_FILENAME).write_text("{ not json")
    encoder = FakeEncoder()
    assert load_known_faces(tmp_path, encoder).names == ["ann"]
    assert encoder.calls == ["ann.jpg"]


def test_cache_of_another_version_is_ignored(tmp_path):
    make_photo(tmp_path, "ann.jpg", "0.1")
    (tmp_path / CACHE_FILENAME).write_text(json.dumps({"version": 0, "entries": {}}))
    encoder = FakeEncoder()
    load_known_faces(tmp_path, encoder)
    assert encoder.calls == ["ann.jpg"]


def test_custom_cache_path(tmp_path):
    photos = tmp_path / "faces"
    photos.mkdir()
    make_photo(photos, "ann.jpg", "0.1")
    cache = tmp_path / "cache.json"
    load_known_faces(photos, FakeEncoder(), cache_path=cache)
    assert cache.exists()
    assert not (photos / CACHE_FILENAME).exists()
