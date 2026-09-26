"""One frame in -> one processed frame out. No window / keyboard code here,
so the same pipeline can be tested on video files.

Right hand pinch -> PIXEL level, left hand pinch -> SWIRL level (both applied to every face).
The swirl is applied first and the pixelation on top of it.
"""
import time

import cv2

from . import hud
from .detectors import Detectors
from .effects import FaceBoxHolder, expand_box, pixelate_region, swirl_region
from .gesture import HandednessStabilizer, LevelController, pick_control_hand
from .settings import Settings


class FacePixelPipeline:
    def __init__(self, settings: Settings, detectors: Detectors):
        self.s = settings
        self.detectors = detectors
        self.pixel_ctrl = LevelController(
            settings.pinch_min_ratio, settings.pinch_max_ratio,
            settings.smoothing_tau, start_level=settings.last_pixel_level)
        self.swirl_ctrl = LevelController(
            settings.pinch_min_ratio, settings.pinch_max_ratio,
            settings.smoothing_tau, start_level=settings.last_swirl_level)
        self.face_holder = FaceBoxHolder(settings.face_hold_sec)
        self.handedness = HandednessStabilizer()
        self.faces = []
        self.hands = []
        self.pixel_hand = None
        self.swirl_hand = None

    def process(self, frame, dt: float):
        """Detect, update both levels and apply the effects. HUD bars are drawn later
        by main.py (after the frame is resized for the window)."""
        s = self.s
        if s.mirror:
            frame = cv2.flip(frame, 1)
        else:
            frame = frame.copy()          # never draw on the camera's buffer
        h, w = frame.shape[:2]

        # 1) Detect
        self.hands, raw_faces = self.detectors.detect(frame)
        self.faces = self.face_holder.update(raw_faces)

        # 2) Gestures -> levels (each one locked when its hand is not visible)
        self.handedness.update(self.hands, (w * w + h * h) ** 0.5, time.perf_counter())
        self.pixel_hand = pick_control_hand(self.hands, s.control_hand, s.mirror)
        self.swirl_hand = pick_control_hand(self.hands, s.swirl_hand, s.mirror)
        for ctrl, hand in ((self.pixel_ctrl, self.pixel_hand), (self.swirl_ctrl, self.swirl_hand)):
            # settings can change live from the Settings window
            ctrl.min_ratio, ctrl.max_ratio, ctrl.tau = s.pinch_min_ratio, s.pinch_max_ratio, s.smoothing_tau
            ctrl.update(hand, dt)
        pixel, swirl = self.pixel_ctrl.level, self.swirl_ctrl.level

        # 3) Effects on every face: swirl first, then pixelate on top of it
        for box in self.faces:
            x1, y1, x2, y2 = expand_box(box, s.face_padding, w, h)
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            radius = max(x2 - x1, y2 - y1) / 2 * s.swirl_radius
            swirl_region(frame, center, radius, swirl * s.max_swirl_deg)
            pixelate_region(frame, (x1, y1, x2, y2), pixel, s.max_blocks, s.min_blocks,
                            noise=s.pixel_noise)

        # 4) Hand overlays
        if s.show_overlays:
            for hand in self.hands:
                if hand is self.pixel_hand:
                    color = hud.CYAN
                elif hand is self.swirl_hand:
                    color = hud.MAGENTA
                else:
                    color = hud.GRAY
                hud.draw_hand(frame, hand, color)
            if self.pixel_hand is not None:
                hud.draw_pinch(frame, self.pixel_hand, pixel, "PIX")
            if self.swirl_hand is not None and self.swirl_hand is not self.pixel_hand:
                hud.draw_pinch(frame, self.swirl_hand, swirl, "SWIRL", hud.BLUE, hud.MAGENTA)
        return frame

    def draw_bars(self, frame):
        hud.draw_level_bar(frame, "PIXEL", self.pixel_ctrl.level, self.pixel_ctrl.locked, slot=0)
        hud.draw_level_bar(frame, "SWIRL", self.swirl_ctrl.level, self.swirl_ctrl.locked, slot=1)

    def reset(self):
        self.pixel_ctrl.reset()
        self.swirl_ctrl.reset()
