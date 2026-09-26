"""Hand + face detection using the MediaPipe Tasks API.

Model files are downloaded automatically into ./models on the first run.
"""
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

MODEL_URLS = {
    "hand_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task",
    "blaze_face_short_range.tflite":
        "https://storage.googleapis.com/mediapipe-models/face_detector/"
        "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
}

# Pairs of landmark indices that form the hand skeleton (21 landmarks).
# Same as vision.HandLandmarksConnections.HAND_CONNECTIONS, written out so it
# also works on older MediaPipe versions.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17),                                   # palm edge
]

WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_MCP = 0, 4, 8, 9


def ensure_model(models_dir: Path, filename: str) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / filename
    if path.exists() and path.stat().st_size > 0:
        return path
    url = MODEL_URLS[filename]
    print(f"[models] Downloading {filename} ...")
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(
            f"Could not download {filename}: {e}\n"
            f"Download it manually from:\n  {url}\nand put it in: {models_dir}"
        ) from e
    print(f"[models] Saved {path}")
    return path


@dataclass
class Hand:
    label: str                     # "Left" / "Right" (the person's real hand when mirror=True)
    score: float
    points: np.ndarray             # (21, 2) pixel coordinates in the full frame


class Detectors:
    def __init__(self, models_dir: Path, detect_width: int = 640):
        self.detect_width = detect_width
        hand_path = ensure_model(models_dir, "hand_landmarker.task")
        face_path = ensure_model(models_dir, "blaze_face_short_range.tflite")

        self.hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(hand_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        ))
        self.faces = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(face_path)),
            running_mode=vision.RunningMode.VIDEO,
            min_detection_confidence=0.5,
        ))
        self._last_ts = -1

    def _timestamp_ms(self) -> int:
        # VIDEO mode requires strictly increasing timestamps.
        ts = int(time.perf_counter() * 1000)
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts
        return ts

    def detect(self, frame_bgr: np.ndarray):
        """Return (hands: list[Hand], faces: list[(x1, y1, x2, y2)]) in full-frame pixels."""
        h, w = frame_bgr.shape[:2]
        scale = min(1.0, self.detect_width / w)
        small = cv2.resize(frame_bgr, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA) if scale < 1.0 else frame_bgr
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = self._timestamp_ms()

        # Hands: landmarks are normalised (0..1) so they map straight back to full size.
        hands = []
        hres = self.hands.detect_for_video(image, ts)
        for lms, handed in zip(hres.hand_landmarks, hres.handedness):
            pts = np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=np.float32)
            hands.append(Hand(label=handed[0].category_name, score=handed[0].score, points=pts))

        # Faces: bounding boxes are in pixels of the small image -> divide by scale.
        faces = []
        fres = self.faces.detect_for_video(image, ts)
        for det in fres.detections:
            b = det.bounding_box
            x1, y1 = b.origin_x / scale, b.origin_y / scale
            x2, y2 = x1 + b.width / scale, y1 + b.height / scale
            faces.append((x1, y1, x2, y2))
        return hands, faces

    def close(self):
        self.hands.close()
        self.faces.close()
