"""Pinch-to-pixelate: the farther your thumb and index finger are apart,
the more pixelated every face on screen becomes.

Run:  python main.py                  (default webcam)
      python main.py --camera 1       (another webcam)
      python main.py --source test.mp4  (a video file, for testing)
"""
import argparse
import time
from pathlib import Path

import cv2

from app import hud
from app.camera import FpsMeter, ThreadedCamera
from app.detectors import Detectors
from app.pipeline import FacePixelPipeline
from app.settings import load_settings, save_settings

ROOT = Path(__file__).resolve().parent
WINDOW = "Pinch to Pixelate"
FPS_PRESETS = {ord("1"): 15, ord("2"): 24, ord("3"): 30, ord("4"): 60}


def parse_args():
    p = argparse.ArgumentParser(description="Pixelate faces with a thumb-index pinch gesture.")
    p.add_argument("--camera", type=int, default=None, help="webcam index (overrides settings.json)")
    p.add_argument("--source", type=str, default=None, help="video file instead of a webcam")
    p.add_argument("--settings", type=Path, default=ROOT / "settings.json")
    return p.parse_args()


def main():
    args = parse_args()
    settings = load_settings(args.settings)
    if args.camera is not None:
        settings.camera_index = args.camera

    source = args.source if args.source else settings.camera_index
    is_webcam = args.source is None

    detectors = Detectors(ROOT / "models", settings.detect_width)
    pipeline = FacePixelPipeline(settings, detectors)

    try:
        camera = ThreadedCamera(source, settings.backend, settings.width, settings.height,
                                settings.target_fps, settings.use_mjpg).start()
    except RuntimeError as e:
        print(f"[error] {e}")
        print("Hints: close other apps using the camera (Zoom, Teams, Camera app), "
              "or try another index:  python main.py --camera 1")
        detectors.close()
        return

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    w, h = camera.actual_size
    cv2.resizeWindow(WINDOW, w, h)

    app_meter = FpsMeter()
    toast_text, toast_until = "", 0.0
    frame_id, last_t = 0, time.perf_counter()

    def toast(text, seconds=2.0):
        nonlocal toast_text, toast_until
        toast_text, toast_until = text, time.perf_counter() + seconds
        print(f"[ui] {text}")

    def set_target_fps(fps):
        fps = max(5, min(120, fps))
        if not is_webcam:
            toast("Target FPS only applies to webcams")
            return
        old = settings.target_fps
        print(f"[ui] Applying target {fps} FPS ...")
        if camera.reopen(fps):
            settings.target_fps = fps
            toast(f"Target FPS -> {fps}")
        elif camera.reopen(old):          # roll back so the app keeps running
            toast(f"Camera refused {fps} FPS, back to {old}")
        else:
            toast("Camera re-open failed! Press Q and restart.")

    try:
        while True:
            new_id, frame = camera.read(frame_id)
            if frame is None:
                continue
            if new_id == frame_id:          # no new frame yet, keep the UI responsive
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                continue
            frame_id = new_id

            now = time.perf_counter()
            dt, last_t = min(now - last_t, 0.2), now
            app_meter.tick(now)

            out = pipeline.process(frame, dt)

            cam_info = f"{camera.backend_name} {camera.actual_fourcc} " \
                       f"{camera.actual_size[0]}x{camera.actual_size[1]}"
            hud.draw_fps(out, app_meter.fps, camera.fps, settings.target_fps, cam_info)
            hud.draw_help(out, settings.show_help)
            if now < toast_until:
                hud.draw_toast(out, toast_text)
            cv2.imshow(WINDOW, out)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key in FPS_PRESETS:
                set_target_fps(FPS_PRESETS[key])
                frame_id = 0
            elif key in (ord("+"), ord("=")):
                set_target_fps(settings.target_fps + 5)
                frame_id = 0
            elif key in (ord("-"), ord("_")):
                set_target_fps(settings.target_fps - 5)
                frame_id = 0
            elif key == ord("m"):
                settings.mirror = not settings.mirror
                toast(f"Mirror {'ON' if settings.mirror else 'OFF'}")
            elif key == ord("o"):
                settings.show_overlays = not settings.show_overlays
                toast(f"Overlays {'ON' if settings.show_overlays else 'OFF'}")
            elif key == ord("h"):
                settings.show_help = not settings.show_help
            elif key == ord("r"):
                pipeline.controller.reset()
                toast("Pixel level reset to 0%")
            elif key == ord("p"):
                if not camera.open_driver_settings():
                    toast("Driver settings need the DSHOW backend (Windows)")

            # Stop when the window's X button is clicked.
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        settings.last_pixel_level = round(pipeline.level, 3)
        save_settings(settings, args.settings)
        camera.stop()
        detectors.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
