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
