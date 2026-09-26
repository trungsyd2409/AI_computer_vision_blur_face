"""Threaded webcam reader tuned for high FPS on Windows.

Why webcams often stay at ~10 FPS on Windows:
  1. The default pixel format is raw YUY2. At 720p/1080p the USB bandwidth
     only allows ~5-10 FPS in YUY2. MJPG (compressed) allows 30+ FPS.
  2. In low light, auto-exposure makes each frame longer (e.g. 1/10 s),
     which caps the FPS no matter what we ask for.
  3. Reading frames in the same loop as heavy processing adds delay.

This class fixes (1) and (3): it asks for MJPG through DirectShow and reads
frames in a background thread, so the main loop always gets the newest frame.
"""
import platform
import threading
import time
from collections import deque

import cv2

IS_WINDOWS = platform.system() == "Windows"

BACKENDS = {
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "any": cv2.CAP_ANY,
}


def fourcc_to_str(value: float) -> str:
    code = int(value)
    chars = [chr((code >> (8 * i)) & 0xFF) for i in range(4)]
    text = "".join(chars).strip("\x00 ")
    return text if text.isprintable() and text else "?"


class FpsMeter:
    """Frames per second over a sliding time window."""

    def __init__(self, window_sec: float = 1.0):
        self.window = window_sec
        self.times = deque()

    def tick(self, now: float | None = None) -> None:
        now = time.perf_counter() if now is None else now
        self.times.append(now)
        while self.times and now - self.times[0] > self.window:
            self.times.popleft()

    @property
    def fps(self) -> float:
        if len(self.times) < 2:
            return 0.0
        span = self.times[-1] - self.times[0]
        return (len(self.times) - 1) / span if span > 0 else 0.0


class ThreadedCamera:
    def __init__(self, source, backend="auto", width=1280, height=720,
                 fps=30, use_mjpg=True):
        self.source = source            # int (webcam index) or str (video file)
        self.backend = backend
        self.width, self.height = width, height
        self.target_fps = fps
        self.use_mjpg = use_mjpg

        self.cap = None
        self.backend_name = "-"
        self.actual_fourcc = "-"
        self.actual_size = (0, 0)
        self.reported_fps = 0.0

        self._lock = threading.Condition()
        self._frame = None
        self._frame_id = 0
        self._running = False
        self._thread = None
        self.meter = FpsMeter()

    # ------------------------------------------------------------------ open
    def _backend_order(self):
        if isinstance(self.source, str):          # video file
            return [("any", cv2.CAP_ANY)]
        if self.backend != "auto":
            return [(self.backend, BACKENDS.get(self.backend, cv2.CAP_ANY))]
        if IS_WINDOWS:
            # DirectShow handles MJPG + FPS requests best on most webcams.
            return [("dshow", cv2.CAP_DSHOW), ("msmf", cv2.CAP_MSMF), ("any", cv2.CAP_ANY)]
        return [("any", cv2.CAP_ANY)]

    def _configure(self, cap):
        # Order matters on DirectShow: pixel format first, then size, then FPS.
        if self.use_mjpg:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # keep latency low (ignored by some drivers)

    def open(self) -> bool:
        for name, api in self._backend_order():
            cap = cv2.VideoCapture(self.source, api)
            if not cap.isOpened():
                cap.release()
                continue
            if not isinstance(self.source, str):
                self._configure(cap)
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                continue
            self.cap = cap
            self.backend_name = name.upper()
            self.actual_fourcc = fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC))
            self.actual_size = (frame.shape[1], frame.shape[0])
            self.reported_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            with self._lock:
                self._frame, self._frame_id = frame, 1
            print(f"[camera] {self.backend_name} {self.actual_size[0]}x{self.actual_size[1]} "
                  f"format={self.actual_fourcc} driver-fps={self.reported_fps:.1f} "
                  f"(target {self.target_fps})")
            return True
        return False

    # --------------------------------------------------------------- thread
    def start(self) -> "ThreadedCamera":
        if self.cap is None and not self.open():
            raise RuntimeError(f"Cannot open camera/video source: {self.source}")
        self.meter = FpsMeter()
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        return self

    def _reader(self):
        is_file = isinstance(self.source, str)
        file_dt = 1.0 / (self.reported_fps or 30.0)
        next_t = time.perf_counter()
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                if is_file:            # loop video files for testing
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.005)
                continue
            if is_file:                # play files at their real speed
                next_t += file_dt
                delay = next_t - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_t = time.perf_counter()
            self.meter.tick()
            with self._lock:
                self._frame = frame
                self._frame_id += 1
                self._lock.notify_all()

    def read(self, last_id: int = 0, timeout: float = 0.5):
        """Return (frame_id, frame). Waits until a frame newer than last_id arrives."""
        with self._lock:
            if self._frame_id <= last_id:
                self._lock.wait(timeout)
            return self._frame_id, self._frame

    @property
    def fps(self) -> float:
        return self.meter.fps

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def reopen(self, fps: int | None = None) -> bool:
        """Re-open the camera with a new target FPS (drivers only apply FPS on open)."""
        if fps is not None:
            self.target_fps = fps
        self.stop()
        try:
            self.start()
            return True
        except RuntimeError as e:
            print(f"[camera] {e}")
            return False

    def open_driver_settings(self) -> bool:
        """Show the webcam driver's own settings dialog (DirectShow on Windows only).
        Useful to turn off 'Low light compensation' / auto exposure, which can cap FPS."""
        if self.cap is None or self.backend_name != "DSHOW":
            return False
        return bool(self.cap.set(cv2.CAP_PROP_SETTINGS, 1))
