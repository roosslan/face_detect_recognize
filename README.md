# face_detect_recognize

Recognizes known people on an RTSP camera stream and writes "person seen" events to a
Redis Stream. It also saves a snapshot whenever something moves. A live preview window with
boxes and names is optional.

```
                 ┌──────────────┐  newest frame   ┌──────────────────┐   events   ┌───────┐
 RTSP camera ──▶ │ reader thread│ ──────────────▶ │ recognition      │ ─────────▶ │ Redis │
                 │ (drops old   │        │        │ thread           │            │Stream │
                 │  frames)     │        │        │ dlib: detect+match│           └───────┘
                 └──────────────┘        ▼        └──────────────────┘
                                 preview window (main thread)
```

## Requirements

- Python 3.12 or newer
- A working `dlib` (installed by `face_recognition`). On Windows this needs CMake and the
  Visual Studio C++ build tools unless a prebuilt wheel is available for your Python version.
- A Redis server

```
pip install -r requirements.txt
```

## Setup

1. Create the `faces/` folder and put **one clear photo per person** into it. The file name
   without the extension is the name that ends up in the events (`rasa.jpg` → `rasa`).
   Supported formats: `.jpg`, `.jpeg`, `.png`. Photos are personal data: keep them out of git
   (for example with an entry in `.git/info/exclude`).
2. Copy the example config and edit it:

   ```
   cp config.example.toml config.toml
   ```

   At minimum set `camera.rtsp_url`. `config.toml` may contain the camera password, so keep it
   out of git as well.

## Usage

```
python main.py                    # run with the preview window, press q to quit
python main.py --headless         # no window, stop with Ctrl+C
python main.py --config other.toml   # default: config.toml in the current folder
python main.py --list             # print the last 20 events from Redis and exit
python main.py --list 50
```

`python -m facerec` works the same way as `python main.py`.

## Building a single exe

On Windows the program can be packed into one `facerec.exe` with PyInstaller:

```
pip install -r requirements-build.txt
pyinstaller facerec.spec
```

The result is `dist/facerec.exe` (about 180 MB: OpenCV, dlib and dlib's face models are inside).
It is built for the Python and Windows version it was built with, so build it on a machine
where the program already runs from source.

To run it, put these next to the exe:

```
facerec.exe
config.toml      copy of config.example.toml with your camera
faces/           one photo per person
```

`capture/` and the encodings cache (`faces/.encodings_cache.json`) are created there too. The
exe always works in its own folder, whichever folder or shortcut it is started from. An explicit
`--config` path is taken relative to where you typed it. All command line options are the same
as for `python main.py`. It is a console program: logs go to the window and Ctrl+C stops it.
Because it is a single file, it unpacks itself to a temporary folder on every start, which
takes a few seconds.

## Configuration

All options are documented in [`config.example.toml`](config.example.toml). Unknown keys and
invalid values are rejected at startup with a clear message.

| Section         | Key                   | Meaning                                                    |
|-----------------|-----------------------|------------------------------------------------------------|
| `[camera]`      | `rtsp_url`            | Stream URL (required); credentials are masked in logs       |
| `[camera]`      | `transport`           | `tcp` (reliable) or `udp` (slightly lower latency)          |
| `[camera]`      | `timeout_sec`         | Open/read timeout; a hung camera triggers a reconnect       |
| `[recognition]` | `tolerance`           | Max face distance counted as a match, lower is stricter     |
| `[recognition]` | `detect_scale`        | Frames are downscaled by this factor before detection; **the main knob for detection distance** (see below) |
| `[recognition]` | `upsample`            | Detector upsampling passes: +1 finds smaller faces but is ~4x slower |
| `[recognition]` | `model`               | `hog` (CPU) or `cnn` (more accurate, needs a GPU-enabled dlib) |
| `[events]`      | `log_cooldown_sec`    | Log the same person at most once per N seconds              |
| `[events]`      | `log_unknown`         | Also log faces that were not recognized (`Unknown`)         |
| `[snapshots]`   | `enabled`, `directory` | Snapshots on motion and where to put them (`capture`)      |
| `[snapshots]`   | `cooldown_sec`, `settle_sec` | Minimum pause between snapshots; how long to wait for a face after motion starts |
| `[snapshots]`   | `motion_area`, `motion_delta` | Motion sensitivity (see below)                    |
| `[snapshots]`   | `annotate`            | Draw boxes and names on the snapshot                        |

### Detection distance

The face detector (dlib HOG) needs a face of about 30 px in the image it runs on (with
`upsample = 1`), so the smallest detectable face in the frame is roughly `30 / detect_scale`
px. How far away that is depends on the stream: the more pixels across the frame, the farther a
face is still large enough. Measured on one CPU with synthetic frames; the distances are
approximate and calibrated on the 60 cm that `detect_scale = 0.25` was observed to reach on a
720x480 stream:

| `detect_scale` | Smallest face | 720x480: distance, time | 1280x720: distance, time |
|----------------|---------------|-------------------------|--------------------------|
| 0.25           | ~120 px       | 0.6 m, 8 ms             | 1.1 m, 23 ms             |
| **0.5 (default)** | ~60 px     | 1.1 m, 33 ms            | 2.0 m, 87 ms             |
| 0.75           | ~40 px        | 1.7 m, 76 ms            | 3.0 m, 195 ms            |
| 1.0            | ~30 px        | 2.2 m, 131 ms           | 3.9 m, 352 ms            |

Face encodings are always computed on the full-resolution frame, not on the downscaled copy
used for detection, because a small face is recognized much more reliably that way.

## Snapshots on motion

When something moves in front of the camera, a snapshot is saved to `capture/`
(`[snapshots]` in the config). The file name is `dd-mm-yyyy-hh-mm-person_name.jpg`, for example
`20-09-2026-14-03-rasa.jpg`; when nobody was recognized the name part is left out:
`20-09-2026-14-03.jpg`. Several recognized people are joined with `+`
(`...-rasa+tima.jpg`), and a second snapshot within the same minute gets ` (2)`, ` (3)`, ...
instead of overwriting the first one.

Recognition needs a moment (a person has to enter the frame and turn to the camera), so after
motion starts the program waits up to `settle_sec` for a known person to be recognized. The
snapshot is taken as soon as somebody is named, or when that time is up without a name. After a
snapshot the next one is taken no sooner than `cooldown_sec` later. Boxes and names are drawn on
the picture unless `annotate = false`.

Motion is detected by comparing each frame with a slowly adapting background: a person who
stops moving fades into it after a couple of seconds, and a change of the whole picture (lights
switched on, exposure or IR-cut change) is not counted as motion. Tune `motion_area` (share of
the frame that must change) and `motion_delta` (colour change per pixel) if the camera
triggers too often or misses movement. The folder is not cleaned up automatically.

## Events

Every event is an entry of the Redis Stream `face_events` (configurable) with two fields:

```
time = "2026-09-12 13:21:00"
name = "rasa"
```

The stream is trimmed to roughly `stream_maxlen` entries. To follow it live:

```
redis-cli XREAD BLOCK 0 STREAMS face_events $
```

If Redis is down at startup the program exits with an error. If it goes down later, failed
writes are logged and retried on the next frame; recognition keeps running.

## Encodings cache

Computing a face encoding from a large photo takes a while, so encodings are cached in
`faces/.encodings_cache.json`. On the next start only new or changed photos (by size and
modification time) are encoded again; removed photos are dropped from the cache. Delete the
file to force a full rebuild.

## Development

```
pip install -r requirements-dev.txt
ruff check .
pytest
```

The tests use fakes for the camera, dlib and Redis, so they need only `numpy` and run without
a camera, a GPU or a Redis server. The same checks run in GitHub Actions on Linux and Windows
(`.github/workflows/ci.yml`).

```
facerec/
  config.py      TOML config: defaults, validation, credential masking
  capture.py     newest-frame RTSP reader with automatic reconnect
  faces.py       known faces, encodings cache, distance matching
  recognizer.py  detection + matching and the background recognition worker
  events.py      Redis Stream writer with per-person cooldown
  motion.py      motion detection (numpy only)
  snapshots.py   snapshots on motion, file naming
  overlay.py     boxes and names drawn on frames
  app.py         command line, preview window
facerec.spec     PyInstaller build recipe for the single exe
tests/           unit tests
old/             the previous prototype, kept for reference only
```
