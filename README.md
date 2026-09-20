# face_detect_recognize

Recognizes known people on an RTSP camera stream and writes "person seen" events to a
Redis Stream. A live preview window with boxes and names is optional.

```
                 ┌──────────────┐  newest frame   ┌──────────────────┐   events   ┌───────┐
 RTSP camera ──▶ │ reader thread│ ──────────────▶ │ recognition      │ ─────────▶ │ Redis │
                 │ (drops old   │        │        │ thread           │            │Stream │
                 │  frames)     │        │        │ dlib: detect+match│           └───────┘
                 └──────────────┘        ▼        └──────────────────┘
                                 preview window (main thread)
```

## Why it does not lag

The camera is read in its own thread that always keeps only the **newest** frame. Slow face
recognition therefore never blocks the stream: the decoder buffer cannot fill up, latency does
not accumulate, and the preview stays smooth. Recognition runs on whatever the newest frame is
at the moment it becomes free. The stream is opened through OpenCV's FFmpeg backend with
low-latency options (`nobuffer`, `low_delay`, TCP or UDP transport) and reconnects
automatically when the signal is lost. GStreamer is not required.

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
python main.py --config other.toml
python main.py --list             # print the last 20 events from Redis and exit
python main.py --list 50
```

`python -m facerec` works the same way as `python main.py`.

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

### Detection distance

The face detector (dlib HOG) needs a face of about 30 px in the image it runs on (with
`upsample = 1`), so the smallest detectable face in the frame is roughly `30 / detect_scale`
px. The distance is then set by the camera's resolution and field of view: the more pixels
across the frame, the farther a face is still large enough. Measured on a 720x480 stream
(synthetic frames, one CPU); the distances are calibrated on the 60 cm that the old
`0.25` setting was observed to reach:

| `detect_scale` / `upsample` | Smallest face found | Detection time | Approx. distance |
|-----------------------------|---------------------|----------------|------------------|
| 0.25 / 1                    | ~120 px             | 8 ms           | 60 cm            |
| 0.5 / 1                     | ~60 px              | 33 ms          | 1.1 m            |
| **0.75 / 1 (default)**      | ~40 px              | 76 ms          | 1.7 m            |
| 1.0 / 1                     | ~32 px              | 131 ms         | 2.2 m            |

Face encodings are always computed on the full-resolution frame, not on the downscaled copy
used for detection, because a small face is recognized much more reliably that way. A higher
camera resolution (main stream instead of a sub-stream) extends the range further.

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
  app.py         command line, preview window
tests/           unit tests
old/             the previous prototype, kept for reference only
```
