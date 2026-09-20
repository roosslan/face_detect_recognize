"""Drawing recognition results on frames."""

from typing import Any

from facerec.recognizer import UNKNOWN, Detection


def draw_detections(canvas: Any, detections: list[Detection]) -> None:
    """Draw boxes and names in place."""
    import cv2

    for det in detections:
        top, right, bottom, left = det.box
        color = (0, 0, 255) if det.name == UNKNOWN else (0, 255, 0)
        cv2.rectangle(canvas, (left, top), (right, bottom), color, 2)
        cv2.rectangle(canvas, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(
            canvas,
            det.name,
            (left + 6, bottom - 6),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (255, 255, 255),
            1,
        )


def annotated(frame: Any, detections: list[Detection]) -> Any:
    """A copy of the frame with the detections drawn on it."""
    canvas = frame.copy()
    draw_detections(canvas, detections)
    return canvas
