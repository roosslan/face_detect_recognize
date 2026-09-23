"""Motion detection against a slowly adapting background (numpy only, no OpenCV)."""

from typing import Any

import numpy as np


class MotionDetector:
    """Feed it frames one by one; `update()` tells whether something moved in this frame.

    The background is a running average, so slow changes (daylight) are absorbed and a person
    who stops moving fades into the background after a couple of seconds.

    Before comparing, the frame is corrected for a camera-wide brightness/white-balance shift
    (auto exposure, auto gain): such a shift changes almost the whole frame, but by a different
    amount in dark and bright areas, so a plain "background - frame" diff treats it as
    widespread motion. Each colour channel is fit to `background * gain + offset` by least
    squares before diffing; real motion is a small part of the frame and barely moves that fit,
    while a moving object still stands out as a local residual afterwards. Two more real-camera
    effects are filtered out separately, because they are not a brightness/gain shift and the
    fit above does not touch them:

    - a night/day switch (the IR-cut filter moving in or out): the picture turns from grey to
      colour or back, which is a change of *kind*, not degree, so it is caught by comparing how
      colourful the frame is, not by the lighting fit;
    - scattered single-block noise (sensor noise, JPEG blocks) that happens to be large enough
      to pass `pixel_delta` in a few unrelated spots: real motion changes a solid patch of
      neighbouring blocks together, so an isolated changed block with no changed neighbour is
      dropped before the changed area is measured.
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
        night_colour: float = 5.0,
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
        saturation: pixel value (0-255) treated as "at the sensor's limit"; a block stuck there
            in both the background and the new frame cannot show a real difference, only the
            camera clipping a bright spot, so it is ignored.
        night_colour: below this, a frame counts as "night" (near greyscale, R, G and B close
            to equal). Crossing this line is treated as an IR-cut switch, not motion.
        """
        self._pixel_delta = pixel_delta
        self._min_area = min_area
        self._max_area = max_area
        self._block = block
        self._alpha = alpha
        self._gain_limits = gain_limits
        self._saturation = saturation
        self._night_colour = night_colour
        self._background: np.ndarray | None = None
        self._peak: np.ndarray | None = None
        self._night: bool | None = None

    def update(self, frame: Any) -> bool:
        gray, peak = self._downscale(frame)  # (height, width, channels)
        if self._background is None or self._background.shape != gray.shape:
            self._background, self._peak = gray, peak
            self._night = self._is_night(gray)
            return False

        night = self._is_night(gray)
        if night != self._night:
            # a grey <-> colour switch is not a brightness change the lighting fit below can
            # model, and not motion either
            self._background, self._peak, self._night = gray, peak, night
            return False

        predicted = self._compensate_lighting(gray)
        residual = gray - predicted
        # a channel counts as changed if it moved on its own, past what the lighting fit
        # already explains; a colourful object of the same overall brightness still shows up,
        # because it is the fit (gain and offset), not brightness itself, being subtracted.
        # Saturation uses the brightest raw pixel in each block, not the block's average: a
        # block that is mostly at the sensor's limit still reads as saturated even if a few
        # darker pixels (an edge, a wire) pull its average down.
        stuck = (self._peak >= self._saturation) & (peak >= self._saturation)
        changed_channel = np.where(stuck, 0.0, np.abs(residual)) > self._pixel_delta
        changed_mask = self._erode(changed_channel.any(axis=-1))
        changed = float(changed_mask.mean())
        if changed > self._max_area:
            self._background, self._peak = gray, peak
            return False
        self._background = predicted + self._alpha * residual
        self._peak = peak
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

    def _is_night(self, gray: np.ndarray) -> bool:
        """True for a near-greyscale frame (R, G and B close to equal): infrared/night mode.
        Frames with fewer than 3 channels are never "night" (there is no colour to lose)."""
        if gray.shape[-1] < 3:
            return False
        b, g, r = gray[..., 0], gray[..., 1], gray[..., 2]
        colour = (np.abs(r - g) + np.abs(g - b) + np.abs(r - b)).mean()
        return bool(colour < self._night_colour)

    @staticmethod
    def _erode(mask: np.ndarray) -> np.ndarray:
        """Drop a changed block that has no changed 4-connected neighbour. Sensor/JPEG noise
        lights up scattered single blocks; real motion lights up a solid patch of them, which
        this leaves untouched (shrunk by one block at the edge, same as any erosion)."""
        padded = np.pad(mask, 1, constant_values=False)
        return (
            padded[1:-1, 1:-1]
            & padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 2:]
        )

    def _downscale(self, frame: Any) -> tuple[np.ndarray, np.ndarray]:
        """Block-average (for comparing) and block-max (for spotting saturation) of the frame."""
        b = self._block
        height, width = frame.shape[0] - frame.shape[0] % b, frame.shape[1] - frame.shape[1] % b
        blocks = frame[:height, :width].reshape(height // b, b, width // b, b, -1)
        return blocks.mean(axis=(1, 3), dtype=np.float32), blocks.max(axis=(1, 3))
