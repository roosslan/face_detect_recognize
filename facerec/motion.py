"""Motion detection against a slowly adapting background (numpy only, no OpenCV)."""

from typing import Any

import numpy as np


class MotionDetector:
    """Feed it frames one by one; `update()` tells whether something moved in this frame.

    The background is a running average, so slow changes (daylight) are absorbed and a person
    who stops moving fades into the background after a couple of seconds.
    """

    def __init__(
        self,
        pixel_delta: float = 25.0,
        min_area: float = 0.005,
        max_area: float = 0.6,
        block: int = 4,
        alpha: float = 0.05,
    ):
        """
        pixel_delta: change (0-255) of any colour channel that makes a pixel count as changed.
        min_area: share of the frame that must change to report motion.
        max_area: a bigger change is a light or exposure switch, not motion: the background
            is reset and nothing is reported.
        block: frames are averaged over block x block pixels first, which removes sensor
            noise and makes the comparison cheap.
        alpha: how fast the background follows the picture (0..1).
        """
        self._pixel_delta = pixel_delta
        self._min_area = min_area
        self._max_area = max_area
        self._block = block
        self._alpha = alpha
        self._background: np.ndarray | None = None

    def update(self, frame: Any) -> bool:
        gray = self._downscale(frame)  # (height, width, channels)
        if self._background is None or self._background.shape != gray.shape:
            self._background = gray
            return False
        # a pixel counts as changed if ANY colour channel changed: comparing plain brightness
        # would miss a colourful object of the same luminance as the background
        difference = np.abs(gray - self._background).max(axis=-1)
        changed = float((difference > self._pixel_delta).mean())
        if changed > self._max_area:
            self._background = gray
            return False
        self._background += self._alpha * (gray - self._background)
        return changed >= self._min_area

    def _downscale(self, frame: Any) -> np.ndarray:
        b = self._block
        height, width = frame.shape[0] - frame.shape[0] % b, frame.shape[1] - frame.shape[1] % b
        blocks = frame[:height, :width].reshape(height // b, b, width // b, b, -1)
        return blocks.mean(axis=(1, 3), dtype=np.float32)
