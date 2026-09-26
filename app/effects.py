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


_rng = np.random.default_rng()


def pixelate_region(frame: np.ndarray, box, level: float, max_blocks: int, min_blocks: int,
                    noise: float = 0.0) -> None:
    """Pixelate frame[box] in place.

    level 0 -> untouched, level 1 -> only `min_blocks` blocks across the face.
    The block count is relative to the face width, so the look stays the same
    whether the face is near or far from the camera.

    noise (0..100): random colour jitter added to every block, re-rolled each
    frame, so the blocks flicker like a glitchy/broken signal. Its strength
    also grows with `level`.
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
    if noise > 0:
        sigma = noise * 0.8 * level                     # noise 100 at level 100% -> sigma 80
        jitter = _rng.normal(0.0, sigma, small.shape)   # one random colour offset per block
        small = np.clip(small.astype(np.float32) + jitter, 0, 255).astype(np.uint8)
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


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def swirl_region(frame: np.ndarray, center, radius: float, angle_deg: float,
                 spin_deg: float = 0.0, strength: float = 1.0) -> None:
    """Twist the pixels inside a circle around `center`, in place (like a whirlpool).

    angle_deg: twist at the centre; it fades to 0 at the circle's edge.
    spin_deg:  extra rotation of the whole disc. Increase it every frame to make
               the whirlpool turn by itself (it is periodic, so it never "winds up").
    strength:  0..1 blend with the original image (fades the effect in near 0%).
    The outer ring of the disc is cross-faded with the original image, so the
    spinning disc has no hard edge.
    """
    if radius < 4 or strength <= 0.001:
        return
    h, w = frame.shape[:2]
    cx, cy = center
    x1, y1 = max(0, int(cx - radius)), max(0, int(cy - radius))
    x2, y2 = min(w, int(cx + radius) + 1), min(h, int(cy + radius) + 1)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return

    # Coordinates of every pixel in the box, relative to the swirl centre.
    ys, xs = np.mgrid[y1:y2, x1:x2].astype(np.float32)
    dx, dy = xs - cx, ys - cy
    dist = np.sqrt(dx * dx + dy * dy)
    t = np.clip(1.0 - dist / radius, 0.0, 1.0)
    theta = np.deg2rad(angle_deg) * t * t + np.deg2rad(spin_deg)   # twist + spin
    cos, sin = np.cos(theta), np.sin(theta)

    # For each output pixel, look up the source pixel rotated by theta.
    map_x = (cx + dx * cos - dy * sin).astype(np.float32)
    map_y = (cy + dx * sin + dy * cos).astype(np.float32)
    swirled = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # Blend weight: 1 inside 75% of the radius, smooth fade to 0 at the edge.
    alpha = _smoothstep((1.0 - dist / radius) / 0.25) * strength
    alpha = alpha[..., None]
    roi = frame[y1:y2, x1:x2].astype(np.float32)
    frame[y1:y2, x1:x2] = (swirled * alpha + roi * (1 - alpha)).astype(np.uint8)
