"""Motion detection against a slowly adapting background (numpy only, no OpenCV)."""

from typing import Any

import numpy as np


class MotionDetector:
    """Feed it frames one by one; `update()` tells whether something moved in this frame.

    The background is a running average, so slow changes (daylight) are absorbed and a person
    who stops moving fades into the background after a couple of seconds.

    Before comparing, the frame is corrected for a camera-wide brightness/white-balance shift
    (auto exposure, auto gain, IR-cut switching): such a shift changes almost the whole frame,
    but by a different amount in dark and bright areas, so a plain "background - frame" diff
    treats it as widespread motion. Each colour channel is fit to `background * gain + offset`
    by least squares before diffing; real motion is a small part of the frame and barely moves
    that fit, while a moving object still stands out as a local residual afterwards.
    """

    def __init__(
        self,
        pixel_delta: float = 25.0,
        min_area: float = 0.005,
        max_area: float = 0.6,
        block: int = 4,
        alpha: float = 0.05,
        gain_limits: tuple[float, float] = (0.2, 5.0),
        saturation: float = 250.0,
    ):
        """
        pixel_delta: change (0-255) of any colour channel that makes a pixel count as changed.
        min_area: share of the frame that must change to report motion.
        max_area: a bigger change is a scene cut or a glitch, not motion or lighting: the
            background is reset and nothing is reported.
        block: frames are averaged over block x block pixels first, which removes sensor
            noise and makes the comparison cheap.
        alpha: how fast the background follows the picture (0..1).
        gain_limits: the per-channel brightness correction is clamped to this range, so a
            near-blank scene (almost no contrast to estimate the correction from) cannot send
            it to an extreme value.
        saturation: pixel value (0-255) treated as "at the sensor's limit"; a channel stuck
            there in both the background and the new frame cannot show a real difference, only
            the camera clipping a bright spot, so it is ignored.
        """
        self._pixel_delta = pixel_delta
        self._min_area = min_area
        self._max_area = max_area
        self._block = block
        self._alpha = alpha
        self._gain_limits = gain_limits
        self._saturation = saturation
        self._background: np.ndarray | None = None

    def update(self, frame: Any) -> bool:
        gray = self._downscale(frame)  # (height, width, channels)
        if self._background is None or self._background.shape != gray.shape:
            self._background = gray
            return False

        predicted = self._compensate_lighting(gray)
        residual = gray - predicted
        # a channel counts as changed if it moved on its own, past what the lighting fit
        # already explains; a colourful object of the same overall brightness still shows up,
        # because it is the fit (gain and offset), not brightness itself, being subtracted
        stuck = (self._background >= self._saturation) & (gray >= self._saturation)
        difference = np.where(stuck, 0.0, np.abs(residual)).max(axis=-1)
        changed = float((difference > self._pixel_delta).mean())
        if changed > self._max_area:
            self._background = gray
            return False
        self._background = predicted + self._alpha * residual
        return changed >= self._min_area

    def _compensate_lighting(self, gray: np.ndarray) -> np.ndarray:
        """Least-squares fit of `gray[..., c] = gain[c] * background[..., c] + offset[c]` for
        each channel: the camera-wide brightness/colour change since the background was last
        updated. A small moving object has little leverage on a fit over the whole frame."""
        background = self._background
        channels = background.shape[-1]
        gain = np.empty(channels, dtype=np.float32)
        offset = np.empty(channels, dtype=np.float32)
        low, high = self._gain_limits
        for c in range(channels):
            x, y = background[..., c].ravel(), gray[..., c].ravel()
            x_mean, y_mean = x.mean(), y.mean()
            spread = float(np.dot(x - x_mean, x - x_mean)) + 1e-6
            slope = float(np.dot(x - x_mean, y - y_mean) / spread)
            gain[c] = min(max(slope, low), high)
            offset[c] = y_mean - gain[c] * x_mean
        return background * gain + offset

    def _downscale(self, frame: Any) -> np.ndarray:
        b = self._block
        height, width = frame.shape[0] - frame.shape[0] % b, frame.shape[1] - frame.shape[1] % b
        blocks = frame[:height, :width].reshape(height // b, b, width // b, b, -1)
        return blocks.mean(axis=(1, 3), dtype=np.float32)
