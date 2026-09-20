"""Known faces: loading from a folder, encoding cache and matching."""

import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
CACHE_VERSION = 1
CACHE_FILENAME = ".encodings_cache.json"

EncodeFn = Callable[[Path], "np.ndarray | None"]


@dataclass(frozen=True)
class KnownFaces:
    names: list[str]
    encodings: np.ndarray  # shape (N, 128)

    def __len__(self) -> int:
        return len(self.names)

    def match(self, encoding: np.ndarray, tolerance: float) -> tuple[str | None, float | None]:
        """Return (name, distance) of the closest known face, or (None, distance) if it is
        farther than `tolerance`. (None, None) when nothing is known."""
        if not self.names:
            return None, None
        distances = np.linalg.norm(self.encodings - encoding, axis=1)
        best = int(np.argmin(distances))
        distance = float(distances[best])
        return (self.names[best] if distance <= tolerance else None), distance


def default_encode(path: Path) -> np.ndarray | None:
    """Encode the first face found on the photo with dlib (via face_recognition)."""
    import face_recognition

    image = face_recognition.load_image_file(str(path))
    encodings = face_recognition.face_encodings(image)
    return encodings[0] if encodings else None


def load_known_faces(
    faces_dir: str | Path,
    encode: EncodeFn = default_encode,
    cache_path: str | Path | None = None,
) -> KnownFaces:
    """Load every image in `faces_dir`; the file name without extension is the person's name.

    Encodings are cached on disk and recomputed only for files that changed
    (compared by size and modification time)."""
    faces_dir = Path(faces_dir)
    cache_path = Path(cache_path) if cache_path else faces_dir / CACHE_FILENAME
    if not faces_dir.is_dir():
        log.warning("Faces folder '%s' does not exist", faces_dir)
        return KnownFaces([], np.empty((0, 128)))

    cache = _read_cache(cache_path)
    fresh: dict[str, dict] = {}
    names: list[str] = []
    encodings: list[np.ndarray] = []
    reused = computed = 0

    # sort by plain string: Path ordering is case-insensitive on Windows only
    for path in sorted(faces_dir.iterdir(), key=lambda p: p.name):
        if path.suffix.lower() not in IMAGE_SUFFIXES or not path.is_file():
            continue
        stat = path.stat()
        signature = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        entry = cache.get(path.name)
        if entry and entry.get("size") == signature["size"] and (
            entry.get("mtime_ns") == signature["mtime_ns"]
        ):
            encoding = np.asarray(entry["encoding"], dtype=np.float64)
            reused += 1
        else:
            try:
                encoding = encode(path)
            except Exception as exc:
                log.warning("Cannot read '%s': %s", path.name, exc)
                continue
            if encoding is None:
                log.warning("No face found in '%s', skipping", path.name)
                continue
            encoding = np.asarray(encoding, dtype=np.float64)
            computed += 1
        fresh[path.name] = {**signature, "encoding": encoding.tolist()}
        names.append(path.stem)
        encodings.append(encoding)

    if fresh != cache:
        _write_cache(cache_path, fresh)
    log.info("Known faces: %d (cached: %d, encoded now: %d)", len(names), reused, computed)
    matrix = np.vstack(encodings) if encodings else np.empty((0, 128))
    return KnownFaces(names, matrix)


def _read_cache(path: Path) -> dict[str, dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") == CACHE_VERSION and isinstance(data.get("entries"), dict):
            return data["entries"]
    except FileNotFoundError:
        pass
    except (OSError, ValueError) as exc:
        log.warning("Ignoring unreadable encodings cache '%s': %s", path, exc)
    return {}


def _write_cache(path: Path, entries: dict[str, dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps({"version": CACHE_VERSION, "entries": entries}), encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        log.warning("Cannot write encodings cache '%s': %s", path, exc)
