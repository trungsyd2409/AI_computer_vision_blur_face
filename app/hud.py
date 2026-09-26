"""On-screen drawing: FPS, pixel level bar, hand skeleton, help panel."""
import cv2
import numpy as np

from .detectors import HAND_CONNECTIONS, INDEX_TIP, THUMB_TIP

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, GRAY = (255, 255, 255), (170, 170, 170)
GREEN, YELLOW, RED, ORANGE = (80, 220, 100), (0, 215, 255), (60, 60, 235), (0, 150, 255)
CYAN = (230, 200, 40)          # right hand (pixel)
MAGENTA = (220, 80, 220)       # left hand (swirl)
BLUE = (255, 140, 40)

HELP_LINES = [
    ("X", "settings window (camera + effect)"),
    ("1 / 2 / 3 / 4", "target FPS 15 / 24 / 30 / 60"),
    ("+ / -", "target FPS +5 / -5"),
    ("P", "camera driver settings (Windows)"),
    ("M", "mirror on / off"),
    ("O", "overlays on / off"),
    ("R", "reset pixel + swirl to 0"),
    ("H", "show / hide this help"),
    ("Q / Esc", "quit (settings are saved)"),
]


def _ui_scale(frame) -> float:
    return max(0.6, frame.shape[0] / 720)


def _panel(frame, x1, y1, x2, y2, alpha=0.55):
    """Dark semi-transparent rectangle behind text."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    frame[y1:y2, x1:x2] = cv2.addWeighted(roi, 1 - alpha, np.zeros_like(roi), alpha, 0)


def draw_fps(frame, app_fps: float, cam_fps: float, target: int, cam_info: str):
    """Big FPS counter in the top-right corner, with camera details under it."""
    s = _ui_scale(frame)
    h, w = frame.shape[:2]
    ratio = app_fps / target if target else 0
    color = GREEN if ratio >= 0.9 else YELLOW if ratio >= 0.5 else RED

    big = f"FPS {app_fps:4.1f}"
    small1 = f"CAM {cam_fps:4.1f} | target {target}"
    small2 = cam_info
    (bw, bh), _ = cv2.getTextSize(big, FONT, 1.0 * s, 2)
    sw = max(cv2.getTextSize(t, FONT, 0.5 * s, 1)[0][0] for t in (small1, small2))
    box_w = max(bw, sw) + int(24 * s)
    pad = int(12 * s)
    x2, y1 = w - pad, pad
    x1 = x2 - box_w
    y2 = y1 + bh + int(62 * s)
    _panel(frame, x1, y1, x2, y2)
    tx = x1 + int(12 * s)
    cv2.putText(frame, big, (tx, y1 + bh + int(10 * s)), FONT, 1.0 * s, color, 2, cv2.LINE_AA)
    cv2.putText(frame, small1, (tx, y1 + bh + int(34 * s)), FONT, 0.5 * s, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, small2, (tx, y1 + bh + int(54 * s)), FONT, 0.5 * s, GRAY, 1, cv2.LINE_AA)


def draw_level_bar(frame, label: str, level: float, locked: bool, slot: int = 0):
    """Horizontal bar in the bottom-left. slot 0 = bottom, slot 1 = the one above it."""
    s = _ui_scale(frame)
    h, w = frame.shape[:2]
    pad = int(12 * s)
    bar_w, bar_h = int(260 * s), int(16 * s)
    box_h = int(60 * s)
    x1, y2 = pad, h - pad - slot * (box_h + int(8 * s))
    y1 = y2 - box_h
    _panel(frame, x1, y1, x1 + bar_w + int(24 * s), y2)

    tx = x1 + int(12 * s)
    cv2.putText(frame, f"{label} {level * 100:3.0f}%", (tx, y1 + int(24 * s)),
                FONT, 0.65 * s, WHITE, 2, cv2.LINE_AA)
    tag, tag_color = ("LOCKED", ORANGE) if locked else ("LIVE", GREEN)
    (tw, _), _ = cv2.getTextSize(tag, FONT, 0.5 * s, 1)
    cv2.putText(frame, tag, (tx + bar_w - tw, y1 + int(24 * s)),
                FONT, 0.5 * s, tag_color, 1, cv2.LINE_AA)

    by1 = y1 + int(34 * s)
    cv2.rectangle(frame, (tx, by1), (tx + bar_w, by1 + bar_h), GRAY, 1)
    fill = int(bar_w * level)
    if fill > 0:
        cv2.rectangle(frame, (tx, by1), (tx + fill, by1 + bar_h), tag_color, -1)


def draw_hand(frame, hand, color=GRAY):
    """21-point hand skeleton. Controlling hands get a colour, ignored hands are grey."""
    s = _ui_scale(frame)
    pts = hand.points.astype(int)
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, tuple(pts[a]), tuple(pts[b]), color, max(1, int(2 * s)), cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, tuple(p), max(2, int(3 * s)), WHITE, -1, cv2.LINE_AA)


def draw_pinch(frame, hand, level: float, label: str, color_lo=GREEN, color_hi=RED):
    """Line between thumb tip and index tip, coloured from color_lo (0%) to color_hi (100%)."""
    s = _ui_scale(frame)
    p1 = tuple(hand.points[THUMB_TIP].astype(int))
    p2 = tuple(hand.points[INDEX_TIP].astype(int))
    color = tuple(int(a + (b - a) * level) for a, b in zip(color_lo, color_hi))
    cv2.line(frame, p1, p2, color, max(2, int(4 * s)), cv2.LINE_AA)
    for p in (p1, p2):
        cv2.circle(frame, p, max(5, int(9 * s)), color, -1, cv2.LINE_AA)
        cv2.circle(frame, p, max(5, int(9 * s)), WHITE, max(1, int(2 * s)), cv2.LINE_AA)
    mid = ((p1[0] + p2[0]) // 2 + int(10 * s), (p1[1] + p2[1]) // 2)
    cv2.putText(frame, f"{label} {level * 100:.0f}%", mid, FONT, 0.6 * s, WHITE, 2, cv2.LINE_AA)


def draw_help(frame, visible: bool):
    s = _ui_scale(frame)
    h, w = frame.shape[:2]
    if not visible:
        text = "H: help"
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.5 * s, 1)
        x, y = w - tw - int(16 * s), h - int(16 * s)
        _panel(frame, x - int(8 * s), y - th - int(8 * s), x + tw + int(8 * s), y + int(8 * s))
        cv2.putText(frame, text, (x, y), FONT, 0.5 * s, WHITE, 1, cv2.LINE_AA)
        return
    line_h = int(24 * s)
    x1, y1 = int(12 * s), int(12 * s)
    width = int(430 * s)
    _panel(frame, x1, y1, x1 + width, y1 + line_h * (len(HELP_LINES) + 1) + int(16 * s), alpha=0.7)
    kx, dx = x1 + int(12 * s), x1 + int(150 * s)
    cv2.putText(frame, "HOTKEYS", (kx, y1 + int(26 * s)), FONT, 0.5 * s, YELLOW, 1, cv2.LINE_AA)
    for i, (key, desc) in enumerate(HELP_LINES, start=1):
        y = y1 + int(26 * s) + i * line_h
        cv2.putText(frame, key, (kx, y), FONT, 0.5 * s, YELLOW, 1, cv2.LINE_AA)
        cv2.putText(frame, desc, (dx, y), FONT, 0.5 * s, WHITE, 1, cv2.LINE_AA)


def draw_toast(frame, text: str):
    """Short message in the top-centre (e.g. 'Target FPS -> 30')."""
    s = _ui_scale(frame)
    h, w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.7 * s, 2)
    x, y = (w - tw) // 2, int(40 * s)
    _panel(frame, x - int(14 * s), y - th - int(12 * s), x + tw + int(14 * s), y + int(12 * s), alpha=0.7)
    cv2.putText(frame, text, (x, y), FONT, 0.7 * s, YELLOW, 2, cv2.LINE_AA)
