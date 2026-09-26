"""One frame in -> one processed frame out. No window / keyboard code here,
so the same pipeline can be tested on video files."""
import time

import cv2

from . import hud
from .detectors import Detectors
from .effects import FaceBoxHolder, expand_box, pixelate_region
from .gesture import HandednessStabilizer, PixelLevelController, pick_control_hand
from .settings import Settings


class FacePixelPipeline:
    def __init__(self, settings: Settings, detectors: Detectors):
        self.s = settings
        self.detectors = detectors
        self.controller = PixelLevelController(
            settings.pinch_min_ratio, settings.pinch_max_ratio,
            settings.smoothing_tau, start_level=settings.last_pixel_level)
        self.face_holder = FaceBoxHolder(settings.face_hold_sec)
        self.handedness = HandednessStabilizer()
        self.faces = []
        self.hands = []
        self.control_hand = None

    def process(self, frame, dt: float):
        if self.s.mirror:
            frame = cv2.flip(frame, 1)
        else:
            frame = frame.copy()          # never draw on the camera's buffer
        h, w = frame.shape[:2]

        # 1) Detect
        self.hands, raw_faces = self.detectors.detect(frame)
        self.faces = self.face_holder.update(raw_faces)

        # 2) Gesture -> level (locked when the control hand is not visible)
        self.handedness.update(self.hands, (w * w + h * h) ** 0.5, time.perf_counter())
        self.control_hand = pick_control_hand(self.hands, self.s.control_hand, self.s.mirror)
        level = self.controller.update(self.control_hand, dt)

        # 3) Pixelate every face
        for box in self.faces:
            pixelate_region(frame, expand_box(box, self.s.face_padding, w, h),
                            level, self.s.max_blocks, self.s.min_blocks)

        # 4) Overlays
        if self.s.show_overlays:
            for hand in self.hands:
                hud.draw_hand(frame, hand, hand is self.control_hand)
            if self.control_hand is not None:
                hud.draw_pinch(frame, self.control_hand, level)
        hud.draw_level_bar(frame, level, self.controller.locked)
        return frame

    @property
    def level(self) -> float:
        return self.controller.level
