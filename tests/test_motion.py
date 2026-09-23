import numpy as np

from facerec.motion import MotionDetector

H, W = 480, 720


def scene(brightness=100, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    frame = np.full((H, W, 3), brightness, dtype=np.float32)
    if noise:
        frame += rng.normal(0, noise, frame.shape)
    return np.clip(frame, 0, 255).astype(np.uint8)


def with_block(frame, x, y, size=60, value=230):
    out = frame.copy()
    out[y : y + size, x : x + size] = value
    return out


def test_first_frame_is_never_motion():
    assert MotionDetector().update(scene()) is False


def test_static_scene_has_no_motion():
    detector = MotionDetector()
    assert not any(detector.update(scene()) for _ in range(20))


def test_sensor_noise_is_not_motion():
    detector = MotionDetector()
    results = [detector.update(scene(noise=6, seed=i)) for i in range(30)]
    assert not any(results)


def test_moving_object_is_motion():
    detector = MotionDetector()
    detector.update(scene())
    assert detector.update(with_block(scene(), 100, 100)) is True


def test_object_that_stops_fades_into_the_background():
    detector = MotionDetector()
    detector.update(scene())
    frame = with_block(scene(), 100, 100)
    assert detector.update(frame) is True
    results = [detector.update(frame) for _ in range(120)]  # it stays where it is
    assert results[-1] is False


def test_colourful_object_with_the_same_brightness_is_motion():
    # (30, 90, 200) averages to ~107, close to the grey background of 120
    detector = MotionDetector()
    detector.update(scene(brightness=120))
    moved = scene(brightness=120)
    moved[100:160, 100:160] = (30, 90, 200)
    assert detector.update(moved) is True


def test_small_change_below_min_area_is_ignored():
    detector = MotionDetector(min_area=0.01)
    detector.update(scene())
    tiny = with_block(scene(), 100, 100, size=16)  # ~0.07% of the frame
    assert detector.update(tiny) is False


def test_gradual_brightness_drift_is_not_motion():
    detector = MotionDetector()
    results = [detector.update(scene(brightness=100 + i * 0.5)) for i in range(40)]
    assert not any(results)


def test_global_light_change_is_not_motion_and_resets_the_background():
    detector = MotionDetector()
    detector.update(scene(brightness=40))
    assert detector.update(scene(brightness=200)) is False  # lights switched on
    assert detector.update(scene(brightness=200)) is False  # the new level is the baseline now
    assert detector.update(with_block(scene(brightness=200), 300, 200, value=20)) is True


def test_frame_size_change_resets_the_background():
    detector = MotionDetector()
    detector.update(scene())
    assert detector.update(np.zeros((240, 320, 3), np.uint8)) is False


def test_grayscale_frames_are_supported():
    detector = MotionDetector()
    gray = np.full((H, W), 100, np.uint8)
    detector.update(gray)
    moved = gray.copy()
    moved[100:160, 100:160] = 230
    assert detector.update(moved) is True


# --- auto exposure / auto gain "breathing": a real-world cause of false positives ---
#
# A camera's own auto exposure or auto gain control brightens and dims the whole picture, but
# not by the same amount everywhere: a dark wall and a bright doorway shift by different
# absolute amounts under the same gain change. A plain "current - background" diff sees that
# as motion spread across a big chunk of the frame. These tests use a two- or three-level scene
# (dark / mid / bright, like a hallway with a lit doorway) instead of a single flat brightness,
# because with only one brightness level any gain looks identical to a flat offset and would
# hide this failure mode.


def corridor(gain=1.0):
    """A dark hallway with a bright doorway on the right, nobody in it."""
    frame = np.full((H, W, 3), 40.0)
    frame[:, W * 2 // 3 :] = 220.0
    return np.clip(frame * gain, 0, 255).astype(np.uint8)


def three_level_scene(gain=1.0):
    """Dark wall, mid-grey floor, bright doorway close to the sensor's limit."""
    frame = np.full((H, W, 3), 40.0)
    frame[:, W // 3 : 2 * W // 3] = 120.0
    frame[:, 2 * W // 3 :] = 220.0
    return np.clip(frame * gain, 0, 255).astype(np.uint8)


def test_camera_exposure_breathing_is_not_motion():
    detector = MotionDetector()
    detector.update(corridor(1.0))
    ramp = [0.5 + 0.8 * i / 29 for i in range(30)]  # exposure drifts down and back up
    assert not any(detector.update(corridor(gain)) for gain in ramp)


def test_motion_is_still_detected_while_exposure_is_breathing():
    detector = MotionDetector()
    detector.update(corridor(1.0))
    ramp = [0.5 + 0.8 * i / 29 for i in range(30)]
    seen = []
    for i, gain in enumerate(ramp):
        frame = corridor(gain)
        if 10 <= i < 20:  # someone walks through while the exposure is still drifting
            left = 30 + (i - 10) * 20
            frame[200:400, left : left + 60] = 90
        seen.append(detector.update(frame))
    assert all(seen[10:20])
    assert not any(seen[:10] + seen[20:])


def test_sudden_light_switch_does_not_look_like_motion():
    detector = MotionDetector()
    detector.update(corridor(1.0))
    assert detector.update(corridor(2.5)) is False  # the light is switched fully on at once
    assert detector.update(corridor(2.5)) is False  # holds at the new level
    assert detector.update(with_block(corridor(2.5), 300, 200, value=20)) is True  # real object


def test_night_day_switch_is_not_motion():
    """The IR-cut filter moving in or out turns the whole picture from near-greyscale to
    colour or back; that is a change of kind, not the gradual brightness/gain drift the fit
    above corrects for, and must not be reported as motion."""
    detector = MotionDetector()
    night = np.full((H, W, 3), 80, dtype=np.uint8)
    day = np.empty((H, W, 3), dtype=np.uint8)
    day[..., 0], day[..., 1], day[..., 2] = 60, 80, 180  # far from grey
    detector.update(night)
    assert detector.update(day) is False


def test_motion_is_detected_right_after_a_night_day_switch():
    detector = MotionDetector()
    night = np.full((H, W, 3), 80, dtype=np.uint8)
    day = np.empty((H, W, 3), dtype=np.uint8)
    day[..., 0], day[..., 1], day[..., 2] = 60, 80, 180
    detector.update(night)
    detector.update(day)  # the switch itself: absorbed, not reported
    assert detector.update(with_block(day, 100, 100, value=10)) is True


def test_scattered_single_block_noise_is_not_motion():
    """Sensor/JPEG noise can push a handful of unrelated blocks past pixel_delta; real motion
    lights up a solid patch of neighbouring blocks, which is what separates the two here."""
    detector = MotionDetector()
    base = scene()
    detector.update(base)
    noisy = base.astype(np.int16)
    for y in range(0, H, 20):
        for x in range(0, W, 20):
            noisy[y : y + 4, x : x + 4] += 50  # one isolated block, well spaced from the rest
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    assert detector.update(noisy) is False


def test_gradual_climb_past_the_sensors_limit_is_not_motion():
    """The doorway (already the brightest thing in view) clips against 255 as exposure keeps
    rising; that clipping alone must not read as motion, and a real object right after still
    must."""
    detector = MotionDetector()
    frame = three_level_scene(1.0)
    for _ in range(40):  # background settles, including the clipped doorway
        detector.update(frame)
    ramp = [1.0 + 0.02 * i for i in range(41)]  # 1.0 -> 1.8, gradually
    assert not any(detector.update(three_level_scene(gain)) for gain in ramp)
    dark_object = three_level_scene(1.8).copy()
    dark_object[200:260, 300:360] = 10
    assert detector.update(dark_object) is True
