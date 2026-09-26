"""Settings window (open / close with the X key).

A second OpenCV window with sliders (trackbars) for:
  - Camera: target FPS, resolution
  - Image:  brightness, contrast, saturation, auto exposure, exposure, gain
  - Effect: pinch sensitivity, smoothing, pixel block sizes, swirl strength
  - View:   window width

Changes apply immediately (FPS / resolution apply ~0.6 s after you stop
dragging, because the camera must be re-opened). Everything is saved to
settings.json when the app exits.
"""
import time

import cv2
import numpy as np

from .camera import IS_WINDOWS, ThreadedCamera
from .settings import Settings

WINDOW = "Settings"
RESOLUTIONS = [(640, 480), (800, 600), (960, 540), (1280, 720), (1600, 900), (1920, 1080)]
REOPEN_DELAY = 0.6      # seconds to wait after the last FPS/resolution change

# Slider label -> key in camera.props. Ranges cover most UVC webcams; if your
# camera uses a smaller range it simply clamps the value (see "cam" read-back).
CAM_SLIDERS = {
    "Brightness": ("brightness", -64, 255),
    "Contrast": ("contrast", 0, 255),
    "Saturation": ("saturation", 0, 255),
    "Gain": ("gain", 0, 255),
}
# Windows drivers use a log2 scale for exposure: -5 means 2^-5 = 1/32 s.
# The slider shows the number without the minus sign: bigger = shorter = darker.
EXPOSURE_LABEL = "Exposure -" if IS_WINDOWS else "Exposure"
EXPOSURE_RANGE = (1, 13) if IS_WINDOWS else (1, 1000)

BG, WHITE, GRAY = (32, 32, 32), (235, 235, 235), (150, 150, 150)
YELLOW, RED, GREEN = (0, 215, 255), (80, 80, 255), (80, 220, 100)


def _noop(_):
    pass


def _nearest_resolution(w, h) -> int:
    return min(range(len(RESOLUTIONS)),
               key=lambda i: abs(RESOLUTIONS[i][0] - w) + abs(RESOLUTIONS[i][1] - h))


def exposure_seconds(value: float) -> float:
    """Exposure value from the driver -> seconds."""
    return 2.0 ** value if IS_WINDOWS else value / 10000.0   # V4L2 uses 100 us units


class SettingsWindow:
    def __init__(self, settings: Settings, camera: ThreadedCamera, is_webcam: bool):
        self.s = settings
        self.cam = camera
        self.is_webcam = is_webcam
        self.is_open = False
        self._last = {}              # slider label -> last position we handled
        self._reopen_at = None       # time when FPS/resolution should be applied
        self.readback = {}           # prop key -> value the driver reports

    # ----------------------------------------------------------- open/close
    def toggle(self) -> str:
        if self.is_open:
            self.close()
            return "Settings closed"
        self.open()
        return "Settings opened (X to close)"

    def open(self):
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(WINDOW, 40 + self.s.display_width + 20, 40)   # right of the camera window
        for label, lo, hi, value in self._initial_sliders():
            cv2.createTrackbar(label, WINDOW, 0, hi, _noop)
            cv2.setTrackbarMin(label, WINDOW, lo)
            cv2.setTrackbarPos(label, WINDOW, int(np.clip(value, lo, hi)))
            self._last[label] = cv2.getTrackbarPos(label, WINDOW)
        self._read_back_all()
        self.is_open = True
        self._draw_info()

    def close(self):
        if self.is_open:
            cv2.destroyWindow(WINDOW)
        self.is_open = False
        self._last.clear()

    def _initial_sliders(self):
        """(label, min, max, start value) for every slider."""
        s, props = self.s, self.cam.props
        sliders = [
            ("FPS", 5, 60, s.target_fps),
            ("Resolution", 0, len(RESOLUTIONS) - 1, _nearest_resolution(s.width, s.height)),
        ]
        for label, (key, lo, hi) in CAM_SLIDERS.items():
            value = props.get(key, self.cam.get_prop(key))
            if value is None or value == -1:
                value = (lo + hi) // 2 if lo >= 0 else 0
            sliders.append((label, lo, hi, round(value)))

        sliders.append(("Auto exposure", 0, 1, props.get("auto_exposure", 1)))
        exp = props.get("exposure", self.cam.get_prop("exposure"))
        if IS_WINDOWS:
            pos = 5 if exp is None or exp >= 0 else round(-exp)
        else:
            pos = 150 if not exp or exp <= 0 else round(exp)
        sliders.append((EXPOSURE_LABEL, *EXPOSURE_RANGE, pos))

        sliders += [
            ("Pinch 0% at", 0, 100, round(s.pinch_min_ratio * 100)),
            ("Pinch 100% at", 20, 300, round(s.pinch_max_ratio * 100)),
            ("Smooth ms", 0, 400, round(s.smoothing_tau * 1000)),
            ("Blocks at 0%", 8, 96, s.max_blocks),
            ("Blocks at 100%", 2, 32, s.min_blocks),
            ("Pixel noise", 0, 100, s.pixel_noise),
            ("Swirl max deg", 0, 1080, s.max_swirl_deg),
            ("Window width", 320, 1920, s.display_width),
        ]
        return sliders

    # ------------------------------------------------------------- per frame
    def update(self) -> str | None:
        """Read the sliders and apply what changed. Returns a toast message or None."""
        if not self.is_open:
            return None
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:   # user clicked the window's X
            self.is_open = False
            self._last.clear()
            return "Settings closed"

        now = time.perf_counter()
        message = None
        for label in list(self._last):
            pos = cv2.getTrackbarPos(label, WINDOW)
            if pos != self._last[label]:
                self._last[label] = pos
                message = self._on_change(label, pos, now) or message

        if self._reopen_at is not None and now >= self._reopen_at:
            self._reopen_at = None
            message = self._apply_fps_resolution()

        self._draw_info()
        return message

    def sync_fps(self, fps: int):
        """Keep the FPS slider in sync when FPS is changed with the 1-4 / +- keys."""
        if self.is_open:
            cv2.setTrackbarPos("FPS", WINDOW, fps)
            self._last["FPS"] = cv2.getTrackbarPos("FPS", WINDOW)

    # -------------------------------------------------------------- changes
    def _set_slider(self, label, pos):
        cv2.setTrackbarPos(label, WINDOW, pos)
        self._last[label] = cv2.getTrackbarPos(label, WINDOW)

    def _on_change(self, label, pos, now) -> str | None:
        s = self.s
        if label in ("FPS", "Resolution"):
            if not self.is_webcam:
                return "FPS / resolution only apply to webcams"
            self._reopen_at = now + REOPEN_DELAY          # wait until you stop dragging
            return None

        if label in CAM_SLIDERS:
            key = CAM_SLIDERS[label][0]
            self.readback[key] = self.cam.set_prop(key, pos)
        elif label == "Auto exposure":
            self.readback["auto_exposure"] = self.cam.set_prop("auto_exposure", pos)
            self.readback["exposure"] = self.cam.get_prop("exposure")
        elif label == EXPOSURE_LABEL:
            value = -pos if IS_WINDOWS else pos
            if self.cam.props.get("auto_exposure", 1):     # moving exposure = manual mode
                self._set_slider("Auto exposure", 0)
                self.cam.props["auto_exposure"] = 0
            self.cam.props["exposure"] = value
            self.readback["auto_exposure"] = self.cam.set_prop("auto_exposure", 0)
            self.readback["exposure"] = self.cam.get_prop("exposure")

        # ---- effect settings (the pipeline reads these every frame)
        elif label == "Pinch 0% at":
            s.pinch_min_ratio = pos / 100
            if s.pinch_max_ratio < s.pinch_min_ratio + 0.2:
                self._set_slider("Pinch 100% at", pos + 20)
                s.pinch_max_ratio = cv2.getTrackbarPos("Pinch 100% at", WINDOW) / 100
        elif label == "Pinch 100% at":
            s.pinch_max_ratio = pos / 100
            if s.pinch_max_ratio < s.pinch_min_ratio + 0.2:
                self._set_slider("Pinch 0% at", max(0, pos - 20))
                s.pinch_min_ratio = cv2.getTrackbarPos("Pinch 0% at", WINDOW) / 100
        elif label == "Smooth ms":
            s.smoothing_tau = pos / 1000
        elif label == "Blocks at 0%":
            s.max_blocks = pos
        elif label == "Blocks at 100%":
            s.min_blocks = pos
        elif label == "Swirl max deg":
            s.max_swirl_deg = pos
        elif label == "Pixel noise":
            s.pixel_noise = pos
        elif label == "Window width":
            s.display_width = pos
        return None

    def _apply_fps_resolution(self) -> str:
        fps = cv2.getTrackbarPos("FPS", WINDOW)
        w, h = RESOLUTIONS[cv2.getTrackbarPos("Resolution", WINDOW)]
        old = (self.s.target_fps, self.s.width, self.s.height)
        if (fps, w, h) == old:
            return f"{w}x{h} @ {fps} FPS (no change)"
        print(f"[settings] Re-opening camera at {w}x{h} @ {fps} FPS ...")
        if self.cam.reopen(fps, w, h):
            self.s.target_fps, self.s.width, self.s.height = fps, w, h
            self._read_back_all()
            aw, ah = self.cam.actual_size
            note = "" if (aw, ah) == (w, h) else f" (camera gave {aw}x{ah})"
            return f"Camera: {w}x{h} @ {fps} FPS{note}"
        self.cam.reopen(*old)                                   # roll back
        self._set_slider("FPS", old[0])
        self._set_slider("Resolution", _nearest_resolution(old[1], old[2]))
        return f"Camera refused {w}x{h} @ {fps}, rolled back"

    def _read_back_all(self):
        for key in ("brightness", "contrast", "saturation", "gain", "auto_exposure", "exposure"):
            self.readback[key] = self.cam.get_prop(key)

    # ------------------------------------------------------------ info panel
    def _draw_info(self):
        img = np.full((374, 560, 3), BG, np.uint8)
        y = 26

        def line(text, color=WHITE, scale=0.5, gap=24):
            nonlocal y
            cv2.putText(img, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
            y += gap

        cam = self.cam
        w, h = RESOLUTIONS[cv2.getTrackbarPos("Resolution", WINDOW)]
        fps = cv2.getTrackbarPos("FPS", WINDOW)
        line("CAMERA", YELLOW)
        line(f"Now: {cam.backend_name} {cam.actual_fourcc} {cam.actual_size[0]}x{cam.actual_size[1]}"
             f"   real {cam.fps:4.1f} FPS")
        pending = "  (applying...)" if self._reopen_at else ""
        line(f"Slider: {w}x{h} @ {fps} FPS{pending}", GRAY)
        if not self.is_webcam:
            line("Video file input: camera sliders are disabled.", RED)

        # Exposure advice: exposure time longer than 1/FPS caps the frame rate.
        auto = self.cam.props.get("auto_exposure", 1)
        if auto:
            line("Auto exposure ON: a dark room can drop FPS to 10.", YELLOW)
        else:
            exp = self.cam.props.get("exposure")
            if exp is not None:
                sec = exposure_seconds(exp)
                max_fps = 1 / sec if sec > 0 else 999
                color = GREEN if max_fps >= fps * 0.95 else RED
                line(f"Exposure {exp:g} = 1/{1 / sec:.0f} s  -> max {min(max_fps, 999):.0f} FPS", color)

        rb = self.readback
        def fmt(key):
            v = rb.get(key)
            return "n/a" if v is None else f"{v:g}"
        line("Camera reports: " + "  ".join(
            f"{k[:4]} {fmt(k)}" for k in ("brightness", "contrast", "saturation", "gain")), GRAY)
        line(f"                exposure {fmt('exposure')}   auto {fmt('auto_exposure')}", GRAY)

        y += 4
        line("EFFECT", YELLOW)
        s = self.s
        line(f"Pinch: 0% at ratio {s.pinch_min_ratio:.2f}, 100% at {s.pinch_max_ratio:.2f}")
        line(f"Smoothing {s.smoothing_tau * 1000:.0f} ms   Blocks {s.max_blocks} -> {s.min_blocks}"
             f"   Noise {s.pixel_noise}")
        line(f"Swirl max {s.max_swirl_deg} deg")
        line(f"Window width {s.display_width} px (height keeps the image ratio)")
        y += 4
        line("X: close   P: driver dialog   Saved to settings.json on exit", GRAY, 0.45)
        cv2.imshow(WINDOW, img)
