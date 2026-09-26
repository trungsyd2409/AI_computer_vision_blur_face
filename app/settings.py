"""Load / save user settings from a JSON file next to main.py."""
import json
from dataclasses import dataclass, asdict, fields
from pathlib import Path


@dataclass
class Settings:
    # --- Camera ---
    camera_index: int = 0
    backend: str = "auto"          # "auto" | "dshow" | "msmf" | "any"
    width: int = 1280
    height: int = 720
    target_fps: int = 30
    use_mjpg: bool = True          # MJPG is the main trick to get 30 FPS on Windows webcams

    # --- View ---
    mirror: bool = True
    show_overlays: bool = True     # skeleton + thumb-index line
    show_help: bool = False

    # --- Gesture -> pixel level ---
    control_hand: str = "Right"    # "Right" | "Left" | "Any"
    pinch_min_ratio: float = 0.25  # pinch ratio at/below this -> 0% pixelation
    pinch_max_ratio: float = 1.50  # pinch ratio at/above this -> 100% pixelation
    smoothing_tau: float = 0.08    # seconds; bigger = smoother but slower response

    # --- Pixelation look ---
    max_blocks: int = 48           # blocks across the face at ~0% (almost sharp)
    min_blocks: int = 4            # blocks across the face at 100% (very blocky)
    face_padding: float = 0.25     # enlarge face box by 25% on each side
    face_hold_sec: float = 0.35    # keep last face box this long if detection drops

    # --- Detection speed ---
    detect_width: int = 640        # frames are downscaled to this width for detection

    # --- State remembered between runs ---
    last_pixel_level: float = 0.0


def load_settings(path: Path) -> Settings:
    s = Settings()
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            valid = {f.name for f in fields(Settings)}
            for k, v in data.items():
                if k in valid:
                    setattr(s, k, v)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[settings] Could not read {path.name} ({e}); using defaults.")
    return s


def save_settings(s: Settings, path: Path) -> None:
    try:
        path.write_text(json.dumps(asdict(s), indent=2), encoding="utf-8")
        print(f"[settings] Saved to {path}")
    except OSError as e:
        print(f"[settings] Could not save settings: {e}")
