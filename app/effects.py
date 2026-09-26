"""Pixelation (mosaic) effect for face regions."""
import time

import cv2
import numpy as np


def expand_box(box, pad: float, frame_w: int, frame_h: int):
    """Grow a face box by `pad` of its size on each side (extra on top for forehead/hair)."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1 -= bw * pad
    x2 += bw * pad
    y1 -= bh * pad * 1.6
    y2 += bh * pad
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(frame_w, int(x2)), min(frame_h, int(y2))
    return x1, y1, x2, y2


def pixelate_region(frame: np.ndarray, box, level: float, max_blocks: int, min_blocks: int) -> None:
    """Pixelate frame[box] in place.

    level 0 -> untouched, level 1 -> only `min_blocks` blocks across the face.
    The block count is relative to the face width, so the look stays the same
    whether the face is near or far from the camera.
    """
    if level <= 0.01:
        return
    x1, y1, x2, y2 = box
    roi = frame[y1:y2, x1:x2]
    h, w = roi.shape[:2]
    if w < 2 or h < 2:
        return
    blocks_x = max_blocks + (min_blocks - max_blocks) * level
    blocks_x = max(1, int(round(blocks_x)))
    blocks_y = max(1, int(round(blocks_x * h / w)))
    small = cv2.resize(roi, (blocks_x, blocks_y), interpolation=cv2.INTER_LINEAR)
    frame[y1:y2, x1:x2] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


class FaceBoxHolder:
    """Keeps the last face boxes for a short time when detection flickers,
    so a face is not suddenly revealed for one frame."""

    def __init__(self, hold_sec: float):
        self.hold_sec = hold_sec
        self.boxes = []
        self.last_seen = 0.0

    def update(self, boxes, now: float | None = None):
        now = time.perf_counter() if now is None else now
        if boxes:
            self.boxes, self.last_seen = boxes, now
        elif now - self.last_seen > self.hold_sec:
            self.boxes = []
        return self.boxes
