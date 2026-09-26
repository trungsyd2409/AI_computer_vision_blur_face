"""Turn the thumb-index distance into a smooth pixelation level (0.0 .. 1.0)."""
import math

import numpy as np

from .detectors import Hand, INDEX_TIP, MIDDLE_MCP, THUMB_TIP, WRIST


class HandednessStabilizer:
    """MediaPipe's Left/Right label can flicker for a frame or two (blur, hand at
    the edge of the image). We follow each hand by its wrist position and keep a
    smoothed probability of it being a right hand, so one bad frame does not
    hand control to the wrong hand."""

    def __init__(self, alpha: float = 0.3, max_jump: float = 0.2, keep_sec: float = 0.3):
        self.alpha = alpha          # weight of the newest frame
        self.max_jump = max_jump    # max wrist movement between frames (fraction of image diagonal)
        self.keep_sec = keep_sec    # forget a hand not seen for this long
        self.tracks = []            # list of dicts: {"pos", "p_right", "t"}

    def update(self, hands: list[Hand], frame_diag: float, now: float) -> None:
        self.tracks = [t for t in self.tracks if now - t["t"] <= self.keep_sec]
        free = list(self.tracks)
        new_tracks = []
        for hand in hands:
            wrist = hand.points[WRIST]
            p_obs = hand.score if hand.label == "Right" else 1.0 - hand.score
            best = min(free, key=lambda t: np.linalg.norm(t["pos"] - wrist), default=None)
            if best is not None and np.linalg.norm(best["pos"] - wrist) < self.max_jump * frame_diag:
                free.remove(best)
                p = (1 - self.alpha) * best["p_right"] + self.alpha * p_obs
            else:
                p = p_obs                                   # a new hand
            hand.label = "Right" if p >= 0.5 else "Left"   # replace with the stable label
            new_tracks.append({"pos": wrist.copy(), "p_right": p, "t": now})
        self.tracks = new_tracks + free


def pick_control_hand(hands: list[Hand], wanted: str, mirror: bool) -> Hand | None:
    """Choose which detected hand controls the effect.

    MediaPipe labels handedness assuming a mirrored (selfie) image. When we do NOT
    mirror the frame, its labels are swapped, so we flip the label we look for.
    """
    if not hands:
        return None
    if wanted == "Any":
        return max(hands, key=lambda h: h.score)
    label = wanted if mirror else ("Left" if wanted == "Right" else "Right")
    matches = [h for h in hands if h.label == label]
    return max(matches, key=lambda h: h.score) if matches else None


def pinch_ratio(hand: Hand) -> float:
    """Thumb-index distance divided by palm size (wrist -> middle knuckle).

    Dividing by palm size makes the value the same whether the hand is near
    or far from the camera.
    """
    p = hand.points
    pinch = np.linalg.norm(p[THUMB_TIP] - p[INDEX_TIP])
    palm = np.linalg.norm(p[WRIST] - p[MIDDLE_MCP])
    return float(pinch / palm) if palm > 1e-6 else 0.0


class PixelLevelController:
    """Keeps the current level. When no control hand is visible the level is locked."""

    def __init__(self, min_ratio: float, max_ratio: float, tau: float, start_level: float = 0.0):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.tau = tau
        self.level = float(np.clip(start_level, 0.0, 1.0))
        self.target = self.level
        self.locked = True
        self.ratio = 0.0

    def update(self, hand: Hand | None, dt: float) -> float:
        if hand is None:
            self.locked = True             # keep the last level as it is
            return self.level
        self.locked = False
        self.ratio = pinch_ratio(hand)
        t = (self.ratio - self.min_ratio) / (self.max_ratio - self.min_ratio)
        self.target = float(np.clip(t, 0.0, 1.0))
        # Frame-rate independent exponential smoothing.
        alpha = 1.0 - math.exp(-dt / self.tau) if self.tau > 0 else 1.0
        self.level += (self.target - self.level) * alpha
        if abs(self.target - self.level) < 1e-3:
            self.level = self.target
        return self.level

    def reset(self):
        self.level = self.target = 0.0
