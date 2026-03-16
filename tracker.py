"""
Video Mouse Tracker
Retroactively tracks mouse cursor movements and clicks from MP4 screen recordings.
Produces heatmaps, click maps, trajectory visualizations, journey maps, and an HTML report.

Usage: python tracker.py <video_file.mp4>
"""

import sys
import os
import json
import time
import base64
import bisect
from pathlib import Path
from dataclasses import dataclass, field, asdict

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CursorPosition:
    frame_idx: int
    timestamp: float
    x: int
    y: int
    confidence: float
    method: str  # "template+motion", "template+flow", "template", "flow", "motion", "hold", "interpolated"


@dataclass
class ClickEvent:
    frame_idx: int
    timestamp: float
    x: int
    y: int
    confidence: float
    kind: str  # "visual_change", "scene_change", "pause_click"


@dataclass
class ScrollEvent:
    frame_idx: int
    timestamp: float
    pixels: int
    direction: str  # "up" or "down"


@dataclass
class Scene:
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    screenshot_idx: int
    click_count: int = 0
    scroll_total: int = 0
    scroll_time_s: float = 0.0
    is_backtrack: bool = False
    similar_scene_idx: int = -1


@dataclass
class TrackingResult:
    video_path: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration: float
    sample_step: int
    positions: list = field(default_factory=list)      # smoothed (used for viz)
    raw_positions: list = field(default_factory=list)  # first-pass detections
    clicks: list = field(default_factory=list)
    scrolls: list = field(default_factory=list)
    scenes: list = field(default_factory=list)
    frames_analyzed: int = 0
    cursor_detected_count: int = 0


# ---------------------------------------------------------------------------
# Cursor detection — template matching with motion validation
# ---------------------------------------------------------------------------

def _frame_change_ratio(gray_a, gray_b, threshold=25):
    """Fraction of pixels that changed between two grayscale frames."""
    diff = cv2.absdiff(gray_a, gray_b)
    return np.count_nonzero(diff > threshold) / diff.size


def _detect_scroll(prev_gray, curr_gray):
    """Detect vertical scrolling via strip cross-correlation. Returns pixel shift or 0."""
    h, w = prev_gray.shape
    strip_h = min(100, h // 10)
    strip_y = h // 2 - strip_h // 2
    x1, x2 = w // 4, 3 * w // 4
    template = prev_gray[strip_y:strip_y + strip_h, x1:x2]
    margin = 200
    sy1 = max(0, strip_y - margin)
    sy2 = min(h, strip_y + strip_h + margin)
    search = curr_gray[sy1:sy2, x1:x2]
    if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
        return 0
    result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    if max_val > 0.85:
        shift = (sy1 + max_loc[1]) - strip_y
        if abs(shift) > 5:
            return shift
    return 0


def _scroll_time_s(scrolls, sample_step, fps, gap_s=0.5):
    """Total seconds spent in scrolling sessions.

    Consecutive scroll events within gap_s of each other are merged into one
    session.  Each session contributes (last_ts - first_ts + sample_interval)
    seconds.  gap_s is intentionally generous to avoid splitting real gestures.
    """
    if not scrolls:
        return 0.0
    sample_interval = sample_step / fps if fps > 0 else 0.0
    timestamps = sorted(s.timestamp for s in scrolls)
    total = 0.0
    session_start = timestamps[0]
    session_end = timestamps[0]
    for t in timestamps[1:]:
        if t - session_end <= gap_s:
            session_end = t
        else:
            total += (session_end - session_start) + sample_interval
            session_start = t
            session_end = t
    total += (session_end - session_start) + sample_interval
    return round(total, 2)


def _find_cursor_in_intersection(intersection, last_pos, y_min=0, max_area=600, max_dim=50):
    """
    Find cursor-sized contour in a three-frame AND mask.
    Returns (x, y, area) or None.
    """
    contours, _ = cv2.findContours(intersection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20 or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w > max_dim or h > max_dim:
            continue
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect > 4:
            continue
        M = cv2.moments(cnt)
        if M["m00"] <= 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        if cy < y_min:
            continue  # skip browser chrome (tab bar)
        candidates.append((cx, cy, area))

    if not candidates:
        return None

    # Animation cluster detection: if 4+ candidates cluster tightly, it's likely
    # a webpage animation (e.g. rotating graphic), not the cursor.  Remove the
    # cluster and prefer any remaining isolated candidate.
    if len(candidates) >= 4:
        # Compute centroid of all candidates
        mx = np.mean([c[0] for c in candidates])
        my = np.mean([c[1] for c in candidates])
        cluster_radius = 250
        in_cluster = [c for c in candidates if np.sqrt((c[0] - mx)**2 + (c[1] - my)**2) < cluster_radius]
        outside = [c for c in candidates if np.sqrt((c[0] - mx)**2 + (c[1] - my)**2) >= cluster_radius]
        # If most candidates are in one cluster, it's an animation zone
        if len(in_cluster) >= 4 and len(in_cluster) >= len(candidates) * 0.6:
            if outside:
                candidates = outside  # use non-animation candidates
            else:
                return None  # all candidates are animation artifacts

    # If we have a last known position, prefer the candidate nearest to it
    if last_pos is not None:
        scored = []
        for c in candidates:
            dist = np.sqrt((c[0] - last_pos[0]) ** 2 + (c[1] - last_pos[1]) ** 2)
            scored.append((c, dist))
        scored.sort(key=lambda s: s[1])
        # Accept if within a reasonable distance
        if scored[0][1] < 400:
            return scored[0][0]

    # No last_pos or nothing near it — return smallest contour (most cursor-like)
    candidates.sort(key=lambda c: c[2])
    return candidates[0]



def _detect_browser_chrome_height(gray_frame):
    """Auto-detect full browser chrome height (tabs + address bar + bookmarks).
    Returns a y_min below which cursor detection should operate."""
    h = gray_frame.shape[0]
    scan_h = min(150, h // 4)
    means = [float(gray_frame[y, :].mean()) for y in range(scan_h)]

    # The chrome-to-content boundary is the FIRST big brightness jump after the
    # address bar region (y>50).  We look for the first jump > 30.
    for y in range(50, scan_h):
        jump = abs(means[y] - means[y - 1])
        if jump > 30:
            return y

    # Fallback: last very-uniform row (browser chrome has low std backgrounds)
    stds = [float(gray_frame[y, :].std()) for y in range(scan_h)]
    last_uniform = 0
    for y in range(scan_h):
        if stds[y] < 3:
            last_uniform = y
    return last_uniform + 2 if last_uniform > 10 else 0


def _detect_url_bar_region(gray_frame):
    """Detect the address bar region (y1, y2) for URL extraction."""
    h = gray_frame.shape[0]
    scan_h = min(120, h // 4)
    means = [float(gray_frame[y, :].mean()) for y in range(scan_h)]
    stds = [float(gray_frame[y, :].std()) for y in range(scan_h)]

    # The address bar is a band of uniform brightness (~75-85) after the tab separator.
    # Find first uniform bright band after y=30 (past the tabs).
    bar_start = bar_end = 0
    in_bar = False
    for y in range(30, scan_h):
        is_uniform = stds[y] < 10 and means[y] > 60
        if not in_bar and is_uniform:
            bar_start = y
            in_bar = True
        elif in_bar and not is_uniform:
            # Allow small gaps (URL text rows have higher std)
            # Check if we re-enter uniform within a few rows
            lookahead = any(stds[yy] < 3 and means[yy] > 60
                           for yy in range(y, min(y + 40, scan_h)))
            if not lookahead:
                bar_end = y
                break
    if in_bar and bar_end == 0:
        bar_end = scan_h

    if bar_start > 0 and bar_end > bar_start:
        return bar_start, bar_end
    return None


@dataclass
class CursorTemplate:
    name: str
    image: np.ndarray      # grayscale template
    mask: np.ndarray        # binary mask or None; None → FFT-accelerated no-mask matching
    hotspot: tuple          # (dx, dy) offset from top-left to logical cursor tip
    scale: float


def _generate_cursor_templates():
    """Generate pixel-accurate templates for common Windows cursors at multiple scales."""
    templates = []

    def _make_arrow(sz=32):
        """Standard Windows arrow cursor — white fill, black border."""
        img = np.zeros((sz, sz), dtype=np.uint8)
        mask = np.zeros((sz, sz), dtype=np.uint8)
        # Arrow polygon (normalized to sz)
        s = sz / 32.0
        pts = np.array([
            [1, 1], [1, 22], [6, 17], [10, 25], [13, 24], [9, 16], [15, 16], [1, 1]
        ], dtype=np.float32) * s
        pts = pts.astype(np.int32)
        cv2.fillPoly(img, [pts], 255)
        cv2.polylines(img, [pts], True, 40, max(1, int(s)))
        cv2.fillPoly(mask, [pts], 255)
        # Dilate mask slightly for matching tolerance
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kern)
        return img, mask, (int(1 * s), int(1 * s))

    def _make_arrow_inverted(sz=32):
        """Dark arrow cursor for light backgrounds."""
        img = np.zeros((sz, sz), dtype=np.uint8)
        mask = np.zeros((sz, sz), dtype=np.uint8)
        s = sz / 32.0
        pts = np.array([
            [1, 1], [1, 22], [6, 17], [10, 25], [13, 24], [9, 16], [15, 16], [1, 1]
        ], dtype=np.float32) * s
        pts = pts.astype(np.int32)
        cv2.fillPoly(img, [pts], 40)
        cv2.polylines(img, [pts], True, 200, max(1, int(s)))
        cv2.fillPoly(mask, [pts], 255)
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kern)
        return img, mask, (int(1 * s), int(1 * s))

    def _make_hand(sz=32):
        """Hand/pointer cursor."""
        img = np.zeros((sz, sz), dtype=np.uint8)
        mask = np.zeros((sz, sz), dtype=np.uint8)
        s = sz / 32.0
        # Simplified hand shape: pointing finger + palm
        pts = np.array([
            [10, 1], [13, 1], [13, 10], [15, 8], [18, 8], [18, 12],
            [20, 10], [23, 10], [23, 14], [25, 12], [27, 12], [27, 18],
            [25, 28], [8, 28], [8, 18], [10, 10]
        ], dtype=np.float32) * s
        pts = pts.astype(np.int32)
        cv2.fillPoly(img, [pts], 255)
        cv2.polylines(img, [pts], True, 40, max(1, int(s)))
        cv2.fillPoly(mask, [pts], 255)
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kern)
        return img, mask, (int(12 * s), int(1 * s))

    def _make_ibeam(sz=32):
        """I-beam text cursor."""
        img = np.zeros((sz, sz), dtype=np.uint8)
        mask = np.zeros((sz, sz), dtype=np.uint8)
        s = sz / 32.0
        cx = int(16 * s)
        # Top serif
        cv2.line(img, (int(12 * s), int(2 * s)), (int(20 * s), int(2 * s)), 200, max(1, int(2 * s)))
        # Bottom serif
        cv2.line(img, (int(12 * s), int(30 * s)), (int(20 * s), int(30 * s)), 200, max(1, int(2 * s)))
        # Vertical bar
        cv2.line(img, (cx, int(2 * s)), (cx, int(30 * s)), 200, max(1, int(1.5 * s)))
        # Build mask from non-zero pixels
        mask[img > 0] = 255
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kern)
        return img, mask, (cx, int(16 * s))

    def _make_crosshair(sz=32):
        """Crosshair cursor."""
        img = np.zeros((sz, sz), dtype=np.uint8)
        mask = np.zeros((sz, sz), dtype=np.uint8)
        s = sz / 32.0
        cx, cy = int(16 * s), int(16 * s)
        thickness = max(1, int(1.5 * s))
        arm = int(10 * s)
        cv2.line(img, (cx - arm, cy), (cx + arm, cy), 200, thickness)
        cv2.line(img, (cx, cy - arm), (cx, cy + arm), 200, thickness)
        mask[img > 0] = 255
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kern)
        return img, mask, (cx, cy)

    def _make_arrow_large_dark(sz=48):
        """Large accessibility arrow — black fill, medium-gray (red in color) border."""
        img = np.zeros((sz, sz), dtype=np.uint8)
        mask = np.zeros((sz, sz), dtype=np.uint8)
        s = sz / 32.0
        pts = np.array([
            [1, 1], [1, 22], [6, 17], [10, 25], [13, 24], [9, 16], [15, 16], [1, 1]
        ], dtype=np.float32) * s
        pts = pts.astype(np.int32)
        # Dark fill with medium-gray border (red → ~76 in grayscale)
        border_thick = max(2, int(2.5 * s))
        cv2.fillPoly(img, [pts], 20)
        cv2.polylines(img, [pts], True, 76, border_thick)
        cv2.fillPoly(mask, [pts], 255)
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kern)
        return img, mask, (int(1 * s), int(1 * s))

    def _make_hand_large_dark(sz=48):
        """Large accessibility hand — black fill, medium-gray border."""
        img = np.zeros((sz, sz), dtype=np.uint8)
        mask = np.zeros((sz, sz), dtype=np.uint8)
        s = sz / 32.0
        pts = np.array([
            [10, 1], [13, 1], [13, 10], [15, 8], [18, 8], [18, 12],
            [20, 10], [23, 10], [23, 14], [25, 12], [27, 12], [27, 18],
            [25, 28], [8, 28], [8, 18], [10, 10]
        ], dtype=np.float32) * s
        pts = pts.astype(np.int32)
        border_thick = max(2, int(2.5 * s))
        cv2.fillPoly(img, [pts], 20)
        cv2.polylines(img, [pts], True, 76, border_thick)
        cv2.fillPoly(mask, [pts], 255)
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kern)
        return img, mask, (int(12 * s), int(1 * s))

    # Standard cursors (white fill, dark border)
    generators = [
        ("arrow", _make_arrow),
        ("arrow_inv", _make_arrow_inverted),
        ("hand", _make_hand),
        ("ibeam", _make_ibeam),
        ("crosshair", _make_crosshair),
    ]

    for scale in [1.0, 1.25, 1.5]:
        sz = int(32 * scale)
        for name, gen_fn in generators:
            img, _mask, hotspot = gen_fn(sz)
            # mask=None → matchTemplate uses FFT (no mask → O(W*H*log(W*H))).
            # TM_CCOEFF_NORMED's mean-centering already de-emphasises the
            # synthetic black background, so accuracy stays high without mask.
            templates.append(CursorTemplate(
                name=name, image=img, mask=None,
                hotspot=hotspot, scale=scale,
            ))

    # Large accessibility cursors (black fill, colored border) at bigger sizes
    large_generators = [
        ("arrow_dark_lg", _make_arrow_large_dark),
        ("hand_dark_lg", _make_hand_large_dark),
    ]
    for scale in [1.0, 1.25, 1.5]:
        sz = int(48 * scale)
        for name, gen_fn in large_generators:
            img, _mask, hotspot = gen_fn(sz)
            templates.append(CursorTemplate(
                name=name, image=img, mask=None,
                hotspot=hotspot, scale=scale,
            ))

    return templates


class CursorKalmanFilter:
    """Constant-velocity Kalman filter for cursor tracking: state = [x, y, vx, vy]."""

    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self._initialized = False

    def predict(self):
        if not self._initialized:
            return None
        pred = self.kf.predict()
        return int(pred[0, 0]), int(pred[1, 0])

    def correct(self, x, y, confidence=0.8):
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        # Scale measurement noise inversely with confidence
        noise = max(0.1, 2.0 - confidence * 2.0)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * noise
        if not self._initialized:
            self.kf.statePost = np.array(
                [[np.float32(x)], [np.float32(y)], [np.float32(0)], [np.float32(0)]],
                dtype=np.float32)
            self._initialized = True
            return
        try:
            self.kf.correct(measurement)
        except cv2.error:
            # Numerical instability — reinitialize at current position
            self.kf.statePost = np.array(
                [[np.float32(x)], [np.float32(y)], [np.float32(0)], [np.float32(0)]],
                dtype=np.float32)
            self.kf.errorCovPost = np.eye(4, dtype=np.float32)

    def get_position(self):
        if not self._initialized:
            return None
        s = self.kf.statePost
        return int(s[0, 0]), int(s[1, 0])


def _compute_search_region(last_pos, kalman_pred, frame_shape, frames_since_detection):
    """Compute a search crop region around predicted cursor location.
    Returns (x1, y1, x2, y2) or None for full-frame search."""
    h, w = frame_shape[:2]
    center = kalman_pred if kalman_pred is not None else last_pos
    if center is None:
        return None  # full-frame search

    base_radius = 300
    expand = min(30 * frames_since_detection, 300)
    radius = min(base_radius + expand, 600)

    cx, cy = center
    x1 = max(0, cx - radius)
    y1 = max(0, cy - radius)
    x2 = min(w, cx + radius)
    y2 = min(h, cy + radius)
    return (x1, y1, x2, y2)


def _detect_cursor_template(gray_frame, templates, search_region=None,
                            risk_zones=None, prev_cursor_type=None, y_min=0):
    """Find cursor via template matching.
    Returns (x, y, confidence, template_name) or None."""
    h, w = gray_frame.shape[:2]

    full_frame = (search_region is None)
    if not full_frame:
        x1, y1, x2, y2 = search_region
        crop = gray_frame[y1:y2, x1:x2]
        offset_x, offset_y = x1, y1
        # Tight crop: masks are fast (small area), use them for precision.
        # Threshold lower because LK/Kalman already placed us near the cursor.
        base_threshold = 0.60
    else:
        crop = gray_frame
        offset_x, offset_y = 0, 0
        # Full frame: higher threshold to suppress false positives everywhere.
        base_threshold = 0.72

    if crop.size == 0:
        return None

    best = None
    best_score = 0

    for tmpl in templates:
        th, tw = tmpl.image.shape[:2]
        if crop.shape[0] < th or crop.shape[1] < tw:
            continue

        # Custom templates (mask != None): always use mask for precision.
        # Built-in templates (mask == None): no mask → FFT path (fast).
        # On full-frame, built-ins without mask need a higher bar because
        # FFT matching is slightly less discriminative than masked matching.
        if tmpl.mask is not None:
            result = cv2.matchTemplate(crop, tmpl.image, cv2.TM_CCOEFF_NORMED,
                                       mask=tmpl.mask)
        else:
            result = cv2.matchTemplate(crop, tmpl.image, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        # Reject inf/nan — produced when region under mask is uniform (no cursor)
        if not np.isfinite(max_val):
            continue

        # Built-in templates without mask are less precise on large crops —
        # apply stricter threshold to compensate for higher false-positive rate.
        effective_threshold = base_threshold
        if tmpl.mask is None and full_frame:
            effective_threshold = max(base_threshold, 0.76)

        # Small type-continuity bonus (reduced to avoid lock-in)
        score = max_val
        if prev_cursor_type is not None and tmpl.name == prev_cursor_type:
            score += 0.02

        if score < effective_threshold:
            continue

        # Convert to frame coordinates
        cx = offset_x + max_loc[0] + tmpl.hotspot[0]
        cy = offset_y + max_loc[1] + tmpl.hotspot[1]

        # Skip browser chrome
        if cy < y_min:
            continue

        # Raise threshold sharply in risk zones (scrollbar, animation regions)
        if risk_zones is not None:
            in_risk = False
            for rz in risk_zones:
                rx1, ry1, rx2, ry2 = rz
                if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                    in_risk = True
                    break
            if in_risk and max_val < 0.82:
                continue

        # Edge energy validation: a real cursor has strong edges against its
        # background. Reject matches in uniformly dark/light regions.
        local_x = max_loc[0]
        local_y = max_loc[1]
        patch = crop[local_y:local_y + th, local_x:local_x + tw]
        if patch.size > 0:
            lap = cv2.Laplacian(patch, cv2.CV_64F)
            edge_energy = lap.var()
            if edge_energy < 80:
                continue
            local_range = int(patch.max()) - int(patch.min())
            if local_range < 40:
                continue

        if score > best_score:
            best_score = score
            best = (cx, cy, float(min(max_val, 1.0)), tmpl.name)

    return best


def _motion_validate(prev_gray, curr_gray, next_gray, candidate_pos, y_min=0, radius=40):
    """Run three-frame AND near candidate_pos. Returns motion confidence 0.0-1.0."""
    if prev_gray is None or next_gray is None:
        return 0.0

    d1 = cv2.absdiff(prev_gray, curr_gray)
    d2 = cv2.absdiff(curr_gray, next_gray)
    _, t1 = cv2.threshold(d1, 25, 255, cv2.THRESH_BINARY)
    _, t2 = cv2.threshold(d2, 25, 255, cv2.THRESH_BINARY)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    t1 = cv2.dilate(t1, kern)
    t2 = cv2.dilate(t2, kern)
    intersection = cv2.bitwise_and(t1, t2)
    intersection = cv2.morphologyEx(intersection, cv2.MORPH_OPEN, kern)

    # Check for motion pixels near the candidate
    cx, cy = candidate_pos
    h, w = intersection.shape[:2]
    x1 = max(0, cx - radius)
    y1 = max(0, cy - radius)
    x2 = min(w, cx + radius)
    y2 = min(h, cy + radius)
    roi = intersection[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0

    motion_pixels = np.count_nonzero(roi)
    # Normalize: ~50 motion pixels = high confidence
    confidence = min(1.0, motion_pixels / 50.0)
    return confidence


def _detect_scrollbar_region(gray_frame):
    """Identify the scrollbar strip at the right edge. Returns (x1, y1, x2, y2) or None."""
    h, w = gray_frame.shape[:2]
    # Scrollbar is typically 15-20px wide at right edge
    strip_w = 20
    strip = gray_frame[:, w - strip_w:]
    # Scrollbar has relatively uniform vertical strips with occasional thumb
    col_std = np.std(strip, axis=0).mean()
    if col_std < 30:  # relatively uniform — likely scrollbar area
        return (w - strip_w, 0, w, h)
    return None


def _detect_animation_zones(frame_buffer, n_samples=12):
    """Detect regions with persistent motion across multiple frame pairs.
    Returns list of (x1, y1, x2, y2) bounding boxes."""
    if len(frame_buffer) < n_samples + 1:
        n_samples = max(1, len(frame_buffer) - 1)

    # Sample evenly across the first portion of the video
    step = max(1, len(frame_buffer) // (n_samples + 1))
    accumulator = None

    for i in range(0, min(n_samples * step, len(frame_buffer) - 1), step):
        gray_a = frame_buffer[i][1]
        gray_b = frame_buffer[i + 1][1]
        diff = cv2.absdiff(gray_a, gray_b)
        _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
        if accumulator is None:
            accumulator = thresh.astype(np.float32)
        else:
            accumulator += thresh.astype(np.float32)

    if accumulator is None:
        return []

    # Regions that change in >40% of sampled pairs are animation zones
    threshold = n_samples * 255 * 0.4
    persistent = (accumulator > threshold).astype(np.uint8) * 255

    # Clean up with morphological operations
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    persistent = cv2.morphologyEx(persistent, cv2.MORPH_CLOSE, kern)

    contours, _ = cv2.findContours(persistent, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    zones = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            x, y, w, h = cv2.boundingRect(cnt)
            zones.append((x, y, x + w, y + h))
    return zones


def _detect_cursor_color(bgr_frame, search_region=None, last_pos=None, y_min=0):
    """Detect cursor by its colored border (red/orange accessibility cursors).
    Works in HSV color space to find distinctive colored cursor outlines.
    Returns (x, y, confidence, 'color') or None."""
    scale = 1.0
    if search_region is not None:
        x1, y1, x2, y2 = search_region
        crop = bgr_frame[y1:y2, x1:x2]
        offset_x, offset_y = x1, y1
    else:
        crop = bgr_frame
        offset_x, offset_y = 0, 0

    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # Red wraps around hue=0: check both ends of the hue spectrum
    # Use broad thresholds — cursor borders can be dark red (S>=50, V>=40)
    mask1 = cv2.inRange(hsv, np.array([0, 50, 40]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([168, 50, 40]), np.array([180, 255, 255]))
    red_mask = mask1 | mask2

    # Clean up noise
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kern)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kern)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        min_area = 80 * (scale ** 2)
        max_area = 5000 * (scale ** 2)
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        min_dim = int(8 * scale)
        max_dim = int(70 * scale)
        if w > max_dim or h > max_dim or w < min_dim or h < min_dim:
            continue
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect > 4:
            continue
        M = cv2.moments(cnt)
        if M["m00"] <= 0:
            continue
        cx = int(M["m10"] / M["m00"] / scale) + offset_x
        cy = int(M["m01"] / M["m00"] / scale) + offset_y
        if cy < y_min:
            continue

        # Check if the interior is dark (black-fill cursor characteristic)
        rx1 = max(0, x + w // 4)
        ry1 = max(0, y + h // 4)
        rx2 = min(crop.shape[1], x + 3 * w // 4)
        ry2 = min(crop.shape[0], y + 3 * h // 4)
        if rx2 > rx1 and ry2 > ry1:
            interior = cv2.cvtColor(crop[ry1:ry2, rx1:rx2], cv2.COLOR_BGR2GRAY)
            mean_val = interior.mean() if interior.size > 0 else 128
            if mean_val > 100:
                continue  # interior not dark enough — probably not a cursor

        # Confidence based on how cursor-shaped the blob is
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        conf = min(0.9, 0.5 + solidity * 0.4)
        candidates.append((cx, cy, conf, area))

    if not candidates:
        return None

    # Prefer candidate nearest to last known position
    if last_pos is not None:
        scored = []
        for c in candidates:
            dist = np.sqrt((c[0] - last_pos[0]) ** 2 + (c[1] - last_pos[1]) ** 2)
            scored.append((c, dist))
        scored.sort(key=lambda s: s[1])
        if scored[0][1] < 400:
            best = scored[0][0]
            return (best[0], best[1], best[2], "color")

    # No last_pos — return best by confidence
    candidates.sort(key=lambda c: -c[2])
    best = candidates[0]
    return (best[0], best[1], best[2], "color")


def _auto_mask_cursor(gray):
    """Strip background from a cursor crop using two strategies in sequence.

    Strategy 1 — Flood fill from corners:
        Seeds from all four corners + edge midpoints.  Only background pixels
        *connected* to the image boundary are removed.  Works even when the
        cursor colour matches the background because interior cursor pixels
        are not reachable from the corners.  Tolerance adapts to JPEG noise.

    Strategy 2 — Edge-based contour fill (fallback):
        Runs Canny edge detection, then fills the largest closed contour(s).
        Handles tight crops where the cursor reaches the image boundary and
        the flood fill sees no clean corner background to sample.

    Both strategies fall back gracefully to a full-white mask (use the whole
    image) if the result still looks wrong.

    Returns a binary mask (255 = cursor pixel, 0 = background).
    """
    h, w = gray.shape[:2]
    kern3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    if h < 8 or w < 8:
        return np.full((h, w), 255, dtype=np.uint8)

    # ------------------------------------------------------------------ #
    # Strategy 1: Flood-fill from corners                                  #
    # ------------------------------------------------------------------ #
    cs = max(2, min(6, h // 5, w // 5))
    corner_pixels = np.concatenate([
        gray[:cs, :cs].ravel(),
        gray[:cs, w - cs:].ravel(),
        gray[h - cs:, :cs].ravel(),
        gray[h - cs:, w - cs:].ravel(),
    ])
    bg_std = max(5.0, float(np.std(corner_pixels)))
    tolerance = int(min(50, bg_std * 2.5 + 20))

    flood = gray.copy()
    bg_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    seeds = [
        (0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1),
        (w // 2, 0), (w // 2, h - 1), (0, h // 2), (w - 1, h // 2),
    ]
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    for sx, sy in seeds:
        if not bg_mask[sy + 1, sx + 1]:
            cv2.floodFill(flood, bg_mask, (sx, sy), 0,
                          loDiff=(tolerance,), upDiff=(tolerance,), flags=flags)

    cursor_mask = cv2.bitwise_not(bg_mask[1:h + 1, 1:w + 1])
    cursor_mask = cv2.morphologyEx(cursor_mask, cv2.MORPH_OPEN, kern3)
    cursor_mask = cv2.morphologyEx(cursor_mask, cv2.MORPH_CLOSE, kern3)

    coverage = np.count_nonzero(cursor_mask) / float(h * w)
    if 0.04 < coverage < 0.90:
        return cursor_mask

    # ------------------------------------------------------------------ #
    # Strategy 2: Edge-based contour fill                                  #
    # ------------------------------------------------------------------ #
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 20, 60)
    kern5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(edges, kern5)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        edge_mask = np.zeros((h, w), dtype=np.uint8)
        # Fill the top-3 contours by area that each cover at least 2% of image
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
            if cv2.contourArea(cnt) / (h * w) > 0.02:
                cv2.fillPoly(edge_mask, [cnt], 255)
        edge_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, kern3)
        coverage = np.count_nonzero(edge_mask) / float(h * w)
        if 0.04 < coverage < 0.92:
            return edge_mask

    # ------------------------------------------------------------------ #
    # Fallback: use the whole image (no background removal)                #
    # ------------------------------------------------------------------ #
    return np.full((h, w), 255, dtype=np.uint8)


def _load_custom_cursor_templates(cursors_dir):
    """Load cursor images from a session Cursors/ folder as additional templates.

    Supports PNG (with optional alpha channel used as mask) and JPG files.
    For images without an alpha channel (JPG, plain PNG), the background is
    stripped automatically using corner-seeded flood fill — no need to provide
    perfect cut-outs.

    Hotspots can be specified in an optional hotspots.txt file alongside the
    images, one entry per line: `filename.png  hotspot_x  hotspot_y`
    If hotspots.txt is absent, all hotspots default to (0, 0) — crop images
    so the cursor tip is at the top-left corner.
    """
    templates = []
    cursors_path = Path(cursors_dir)
    if not cursors_path.is_dir():
        return templates

    # Parse optional hotspots file
    hotspot_map = {}
    hs_file = cursors_path / "hotspots.txt"
    if hs_file.exists():
        for line in hs_file.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) == 3:
                try:
                    hotspot_map[parts[0]] = (int(parts[1]), int(parts[2]))
                except ValueError:
                    pass

    for img_path in sorted(cursors_path.glob("*")):
        if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp"):
            continue
        if img_path.name.startswith("_"):  # skip auto-generated previews
            continue
        raw = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            continue

        if raw.ndim == 3 and raw.shape[2] == 4:
            # RGBA PNG — alpha channel is already a clean mask, use it directly
            alpha = raw[:, :, 3]
            gray = cv2.cvtColor(raw[:, :, :3], cv2.COLOR_BGR2GRAY)
            mask = np.where(alpha > 10, np.uint8(255), np.uint8(0))
        elif raw.ndim == 3:
            # RGB (JPG or plain PNG) — auto-strip background
            gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
            mask = _auto_mask_cursor(gray)
        else:
            # Already grayscale
            gray = raw
            mask = _auto_mask_cursor(gray)

        # Slight mask dilation for matching tolerance
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kern)

        hotspot = hotspot_map.get(img_path.name, (0, 0))
        templates.append(CursorTemplate(
            name=f"custom_{img_path.stem}",
            image=gray,
            mask=mask,
            hotspot=hotspot,
            scale=1.0,
        ))
        coverage = np.count_nonzero(mask) / float(mask.size) * 100
        method = "alpha" if (raw.ndim == 3 and raw.shape[2] == 4) else (
                 "flood" if coverage < 95 else "fallback")
        print(f"    {img_path.name}: mask covers {coverage:.0f}% ({method})")
        # Save masked preview so you can visually verify the strip worked
        preview = cv2.bitwise_and(gray, gray, mask=mask)
        preview_path = img_path.parent / f"_mask_{img_path.stem}.png"
        cv2.imwrite(str(preview_path), preview)

    return templates


EXCLUSION_RADIUS = 120  # px — radius of a right-click exclusion zone


def _save_anchors(anchor_json, anchors, exclusions):
    data = {
        "anchors":    [{"frame_idx": fi, "x": x, "y": y} for fi, x, y in anchors],
        "exclusions": [{"cx": cx, "cy": cy, "r": r} for cx, cy, r in exclusions],
    }
    try:
        with open(anchor_json, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Saved {len(anchors)} anchor(s) + {len(exclusions)} exclusion(s) to {Path(anchor_json).name}")
    except Exception as e:
        print(f"  Warning: could not save anchors ({e})")


def _collect_anchors(video_path, n_samples=6):
    """
    Interactive anchor collection.

    Opens N frames spread evenly through the video and asks the user to click
    where the cursor is.  RIGHT-CLICK on any false-positive hotspot (spinner,
    scrollbar, etc.) to mark a permanent exclusion zone — the tracker will
    never accept a detection inside that circle for the whole video.

    Anchors + exclusion zones are cached to ``{video_stem}_anchors.json`` next
    to the video file so that re-runs skip this step automatically.
    Delete the JSON to re-annotate.

    Controls:
        LEFT CLICK    — mark cursor position (click again to adjust)
        RIGHT CLICK   — add exclusion zone (120px radius, persists all frames)
        SPACE/ENTER   — confirm and advance (or skip if cursor not visible)
        ESC           — skip this frame
        Q             — stop early, save what has been confirmed so far

    Returns (anchors, exclusions):
        anchors    — list of (frame_idx, x, y)
        exclusions — list of (cx, cy, radius)
    """
    video_path = str(video_path)
    video_stem = Path(video_path).stem[:50]
    anchor_json = Path(video_path).parent / f"{video_stem}_anchors.json"

    # Load cached data if it exists
    if anchor_json.exists():
        try:
            with open(anchor_json, "r") as f:
                data = json.load(f)
            # Support both old list format and new dict format
            if isinstance(data, list):
                anchors    = [(int(a["frame_idx"]), int(a["x"]), int(a["y"])) for a in data]
                exclusions = []
            else:
                anchors    = [(int(a["frame_idx"]), int(a["x"]), int(a["y"])) for a in data.get("anchors", [])]
                exclusions = [(int(e["cx"]), int(e["cy"]), int(e["r"])) for e in data.get("exclusions", [])]
            print(f"  Loaded {len(anchors)} anchor(s) + {len(exclusions)} exclusion(s) from {anchor_json.name}")
            print(f"  (Delete {anchor_json.name} to re-annotate)")
            return anchors, exclusions
        except Exception as e:
            print(f"  Warning: could not load anchor cache ({e}); re-annotating")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("  Warning: cannot open video for anchor collection")
        return [], []

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Pick N evenly-spaced frame indices, skipping first/last 5%
    margin = max(1, int(total * 0.05))
    usable = max(1, total - 2 * margin)
    if n_samples <= 1:
        indices = [margin + usable // 2]
    else:
        indices = sorted(set(
            margin + int(usable * k / (n_samples - 1))
            for k in range(n_samples)
        ))

    anchors    = []
    exclusions = []   # (cx, cy, radius) — global, persists across all frames
    click_pos  = [None]

    # Scale the image to fit within 1280x800 for display, preserving aspect ratio.
    # Use WINDOW_AUTOSIZE so the window is exactly this size — mouse coordinates
    # are then deterministically in display space and we scale them back to image
    # space in the callback.  WINDOW_NORMAL with resizeWindow is unreliable:
    # mouse coords can be in either display or image space depending on platform.
    display_scale = min(1.0, 1280 / width, 800 / height)
    disp_w = max(1, int(width  * display_scale))
    disp_h = max(1, int(height * display_scale))

    def _on_mouse(event, x, y, flags, param):
        # Convert display coords → image coords
        ix = int(round(x / display_scale))
        iy = int(round(y / display_scale))
        if event == cv2.EVENT_LBUTTONDOWN:
            click_pos[0] = (ix, iy)
        elif event == cv2.EVENT_RBUTTONDOWN:
            # If clicking inside an existing zone, remove it; otherwise add one
            for idx, (ex, ey, er) in enumerate(exclusions):
                if np.hypot(ix - ex, iy - ey) <= er:
                    exclusions.pop(idx)
                    print(f"  Exclusion zone removed at ({ex}, {ey})")
                    return
            exclusions.append((ix, iy, EXCLUSION_RADIUS))
            print(f"  Exclusion zone added at ({ix}, {iy}) r={EXCLUSION_RADIUS}px")

    win_name = "Anchor Collection"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win_name, _on_mouse)

    print(f"\n  === Anchor Collection ({n_samples} frames) ===")
    print(f"  LEFT CLICK         — mark cursor position")
    print(f"  RIGHT CLICK        — add exclusion zone  |  right-click inside one = remove it")
    print(f"  Z                  — undo last exclusion zone")
    print(f"  SPACE/ENTER        — confirm and advance  (no click = cursor not visible, skip)")
    print(f"  ESC                — skip this frame")
    print(f"  Q                  — stop early\n")

    quit_early = False
    for frame_num, frame_idx in enumerate(indices, 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, bgr = cap.read()
        if not ret or bgr is None:
            continue

        click_pos[0] = None
        timestamp = frame_idx / fps if fps > 0 else 0

        while True:
            display = bgr.copy()

            # Draw all current exclusion zones (red circles, semi-transparent fill)
            overlay = display.copy()
            for ex, ey, er in exclusions:
                cv2.circle(overlay, (ex, ey), er, (0, 0, 180), -1)
            cv2.addWeighted(overlay, 0.25, display, 0.75, 0, display)
            for ex, ey, er in exclusions:
                cv2.circle(display, (ex, ey), er, (0, 0, 255), 2)
                cv2.putText(display, "X", (ex - 6, ey + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

            # Instruction bar
            cv2.rectangle(display, (0, 0), (width, 32), (30, 30, 30), -1)
            excl_note = f"  [{len(exclusions)} excl]" if exclusions else ""
            info = (f"Frame {frame_num}/{len(indices)}  t={timestamp:.1f}s{excl_note}  |"
                    f"  L=cursor  R=add/remove excl  Z=undo excl  SPACE=confirm/skip  Q=quit")
            cv2.putText(display, info, (6, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (240, 240, 240), 1, cv2.LINE_AA)

            # Draw confirmed cursor click (gold crosshair)
            if click_pos[0] is not None:
                cx_m, cy_m = click_pos[0]
                cv2.circle(display, (cx_m, cy_m), 14, (0, 215, 255), 2)
                cv2.line(display, (cx_m - 20, cy_m), (cx_m + 20, cy_m), (0, 215, 255), 1)
                cv2.line(display, (cx_m, cy_m - 20), (cx_m, cy_m + 20), (0, 215, 255), 1)
                cv2.putText(display, f"({cx_m},{cy_m})", (cx_m + 16, cy_m - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1, cv2.LINE_AA)

            # Show downscaled image — all drawing is done at full image coords
            # above, so resizing here keeps annotations correctly positioned
            if display_scale < 1.0:
                cv2.imshow(win_name, cv2.resize(display, (disp_w, disp_h)))
            else:
                cv2.imshow(win_name, display)
            key = cv2.waitKey(30) & 0xFF

            if key in (ord('q'), ord('Q')):
                quit_early = True
                break
            elif key in (ord('z'), ord('Z')):  # undo last exclusion zone
                if exclusions:
                    removed = exclusions.pop()
                    print(f"  Exclusion zone undone at ({removed[0]}, {removed[1]})")
            elif key in (32, 13):  # SPACE or ENTER
                if click_pos[0] is not None:
                    ax, ay = click_pos[0]
                    anchors.append((frame_idx, ax, ay))
                    print(f"  Anchor {frame_num}: frame {frame_idx} ({timestamp:.1f}s) -> ({ax}, {ay})")
                else:
                    print(f"  Frame {frame_num}: no cursor visible — skipped")
                break
            elif key == 27:  # ESC — skip
                print(f"  Frame {frame_num}: skipped")
                break

        if quit_early:
            break

    cap.release()
    cv2.destroyAllWindows()

    _save_anchors(anchor_json, anchors, exclusions)
    print()
    return anchors, exclusions


def track_cursor(video_path, sample_step=2, progress_callback=None, cursors_dir=None, anchors=None, exclusions=None):
    """
    Track cursor via template matching + LK optical flow tracking.
    Primary:   template matching anchors the cursor position every N frames.
    Secondary: Lucas-Kanade optical flow fills gaps between template detections.
    Tertiary:  three-frame AND motion validates and boosts confidence.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    result = TrackingResult(
        video_path=video_path, width=width, height=height,
        fps=fps, total_frames=total_frames, duration=duration,
        sample_step=sample_step,
    )

    # Read all sampled frames into (frame_idx, gray) tuples.
    # BGR is NOT stored — at 1440x1080 a single BGR frame is ~4.5 MB; buffering
    # all frames of a 6-min video would require ~50 GB of RAM and cause severe
    # swapping. Color detection reads BGR on-demand via a secondary cap instead.
    frame_buffer = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_buffer.append((frame_idx, gray))
        frame_idx += 1
    cap.release()

    # Auto-detect browser chrome height to exclude tabs + address bar from cursor detection
    y_min = _detect_browser_chrome_height(frame_buffer[0][1]) if frame_buffer else 0
    url_region = _detect_url_bar_region(frame_buffer[0][1]) if frame_buffer else None

    # Probe whether this video has a colored (red) accessibility cursor.
    # Reads frames directly from the video (not the gray buffer) to avoid
    # keeping BGR in RAM. Uses a fresh VideoCapture for the probe pass.
    has_color_cursor = False
    if total_frames > 40 * sample_step:
        color_hits = 0
        cap_probe = cv2.VideoCapture(video_path)
        probe_step = max(sample_step, int(total_frames) // 80)
        for pidx in range(10 * sample_step,
                          min(int(total_frames), 800 * sample_step),
                          probe_step):
            cap_probe.set(cv2.CAP_PROP_POS_FRAMES, pidx)
            ret_p, probe_bgr = cap_probe.read()
            if ret_p and probe_bgr is not None:
                if _detect_cursor_color(probe_bgr, y_min=y_min) is not None:
                    color_hits += 1
                    if color_hits >= 5:
                        break
        cap_probe.release()
        if color_hits >= 5:
            has_color_cursor = True
            print(f"  Detected colored (accessibility) cursor — enabling color detection")
    result.url_bar_region = url_region  # store for scene labeling
    print(f"  Loaded {len(frame_buffer)} sampled frames (browser chrome cutoff: y>{y_min})")

    # Build anchor map: maps buffer index i -> (x, y) for user-provided anchors.
    # Each anchor frame_idx is matched to the nearest sampled buffer entry.
    anchor_map = {}
    sorted_anchor_idxs = []
    if anchors:
        buf_fidxs = np.array([fb[0] for fb in frame_buffer], dtype=np.int64)
        for a_frame_idx, ax, ay in anchors:
            buf_i = int(np.argmin(np.abs(buf_fidxs - a_frame_idx)))
            anchor_map[buf_i] = (ax, ay)
        sorted_anchor_idxs = sorted(anchor_map.keys())
        print(f"  Anchors: {len(anchor_map)} hard re-seeding point(s) will be applied during tracking")

    # Generate cursor templates (built-in synthetic shapes)
    templates = _generate_cursor_templates()

    # Prepend custom templates from session Cursors/ folder (higher priority)
    if cursors_dir:
        custom = _load_custom_cursor_templates(cursors_dir)
        if custom:
            templates = custom + templates
            print(f"  Loaded {len(custom)} custom cursor template(s) from {Path(cursors_dir).name}/")

    # Lucas-Kanade optical flow parameters
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    )

    # Initialize Kalman filter
    kalman = CursorKalmanFilter()

    # Detect risk zones (scrollbar + animation regions)
    risk_zones = []
    scrollbar = _detect_scrollbar_region(frame_buffer[0][1]) if frame_buffer else None
    if scrollbar is not None:
        risk_zones.append(scrollbar)
    animation_zones = _detect_animation_zones(frame_buffer) if len(frame_buffer) > 2 else []
    risk_zones.extend(animation_zones)
    if risk_zones:
        print(f"  Detected {len(risk_zones)} risk zone(s) (scrollbar + animations)")

    # Pre-scan for scroll events so we can exclude nearby frames from color
    # detection (scrolling red text/links cause false positives)
    scroll_frame_set = set()
    SCROLL_EXCLUSION_RADIUS = 5  # exclude +/- 5 frames around each scroll
    for si in range(1, len(frame_buffer)):
        prev_g = frame_buffer[si - 1][1]
        curr_g = frame_buffer[si][1]
        ch = _frame_change_ratio(prev_g, curr_g)
        if ch > 0.03:
            sp = _detect_scroll(prev_g, curr_g)
            if sp != 0:
                scroll_idx = frame_buffer[si][0]
                for offset in range(-SCROLL_EXCLUSION_RADIUS, SCROLL_EXCLUSION_RADIUS + 1):
                    scroll_frame_set.add(scroll_idx + offset)

    last_pos = None
    last_confident_pos = None  # last position with conf >= 0.5
    prev_cursor_type = None
    frames_since_detection = 0
    frames_at_same_pos = 0  # how many consecutive frames at ~same position
    detected = 0
    scrolls = []
    lk_point = None  # np.float32 shape (1,1,2) — current LK tracked cursor tip

    # Secondary cap for on-demand BGR reads (color cursor detection only).
    # Seeking is fast enough since color detection fires infrequently.
    cap_color = cv2.VideoCapture(video_path) if has_color_cursor else None

    for i in range(len(frame_buffer)):
        curr_idx, curr_gray = frame_buffer[i]
        timestamp = curr_idx / fps if fps > 0 else 0

        # --- Scroll detection (always check against previous frame) ---
        if i > 0:
            prev_gray = frame_buffer[i - 1][1]
            change = _frame_change_ratio(prev_gray, curr_gray)

            if change > 0.03:
                scroll_px = _detect_scroll(prev_gray, curr_gray)
                if scroll_px != 0:
                    scrolls.append(ScrollEvent(
                        frame_idx=curr_idx,
                        timestamp=round(timestamp, 3),
                        pixels=abs(scroll_px),
                        direction="down" if scroll_px > 0 else "up",
                    ))

        # --- Anchor override ---
        # User-confirmed positions are injected as hard re-seeding points.
        # They bypass all automatic detection, reset LK/Kalman, and become the
        # new baseline for subsequent tracking.
        if i in anchor_map:
            ax, ay = anchor_map[i]
            result.positions.append(CursorPosition(
                frame_idx=curr_idx, timestamp=round(timestamp, 3),
                x=ax, y=ay, confidence=1.00, method="anchor",
            ))
            detected += 1
            last_pos = (ax, ay)
            last_confident_pos = (ax, ay)
            frames_since_detection = 0
            frames_at_same_pos = 0
            lk_point = np.array([[[float(ax), float(ay)]]], dtype=np.float32)
            kalman.correct(ax, ay, 1.00)
            if progress_callback and i % 100 == 0:
                progress_callback(curr_idx, total_frames)
            continue

        # --- Cursor detection ---
        if i > 0:
            prev_gray = frame_buffer[i - 1][1]
            change = _frame_change_ratio(prev_gray, curr_gray)
        else:
            change = 1.0

        method = "hold"
        cx, cy = None, None
        conf = 0.0
        template_conf = 0.0
        template_name = None
        color_conf = 0.0
        motion_conf = 0.0
        lk_conf = 0.0
        kalman_pred = kalman.predict()  # advance Kalman state once per frame

        busy_frame = (change > 0.05)
        # During busy frames, only accept template matches that clear a higher bar.
        # LK flow handles the gaps instead of demanding higher template confidence.
        busy_min_conf = 0.65

        # --- LK optical flow tracking ---
        # Track the cursor tip between template detections using pyramidal LK.
        # Forward-backward error check validates tracking quality each frame.
        lk_cx, lk_cy = None, None
        if lk_point is not None and i > 0:
            prev_gray_lk = frame_buffer[i - 1][1]
            new_pt, st, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray_lk, curr_gray, lk_point, None, **lk_params)
            if st is not None and st[0][0]:
                back_pt, back_st, _ = cv2.calcOpticalFlowPyrLK(
                    curr_gray, prev_gray_lk, new_pt, None, **lk_params)
                if back_st is not None and back_st[0][0]:
                    fb_err = float(np.linalg.norm(
                        lk_point.ravel() - back_pt.ravel()))
                    if fb_err < 2.5:
                        lk_cx = int(round(float(new_pt[0, 0, 0])))
                        lk_cy = int(round(float(new_pt[0, 0, 1])))
                        lk_conf = max(0.50, 0.70 - fb_err * 0.08)
                        lk_point = new_pt
                    else:
                        lk_point = None  # tracking drifted — reset
                else:
                    lk_point = None
            else:
                lk_point = None

        # Validate LK position (must be inside frame and below browser chrome)
        if lk_cx is not None and (
                lk_cy < y_min or not (0 <= lk_cx < width) or not (0 <= lk_cy < height)):
            lk_cx, lk_cy, lk_conf = None, None, 0.0
            lk_point = None

        # Reject LK if it drifted into a user exclusion zone
        if lk_cx is not None and exclusions:
            for ex, ey, er in exclusions:
                if np.hypot(lk_cx - ex, lk_cy - ey) <= er:
                    lk_cx, lk_cy, lk_conf = None, None, 0.0
                    lk_point = None
                    break

        # 1. Compute search region.
        # When LK is active, search a tight window around its prediction — this
        # makes template matching faster and reduces false positives from a wide
        # search.  Fall back to normal Kalman / busy-frame logic otherwise.
        # Full-frame search is expensive even with FFT (~360ms at 1440x1080).
        # Only force it when stuck, or every 150 sampled frames (~10s of video).
        # LK + Kalman prevent drift from accumulating between re-anchors.
        force_full_search = (frames_at_same_pos > 15) or (i % 150 == 0)
        if force_full_search:
            search_region = None  # full frame (periodic re-anchor)
        elif lk_cx is not None:
            fh, fw = curr_gray.shape[:2]
            r = 120  # tight: LK gives a precise prior
            search_region = (max(0, lk_cx - r), max(0, lk_cy - r),
                             min(fw, lk_cx + r), min(fh, lk_cy + r))
        elif busy_frame and last_pos is not None:
            lx, ly = last_pos
            fh, fw = curr_gray.shape[:2]
            r = 450
            search_region = (max(0, lx - r), max(0, ly - r),
                             min(fw, lx + r), min(fh, ly + r))
        else:
            search_region = _compute_search_region(
                last_pos, kalman_pred, curr_gray.shape, frames_since_detection)

        # 1b. Anchor bounding-box constraint.
        # When this frame sits between two confirmed user anchors, intersect the
        # search region with a box that encompasses both anchor positions + margin.
        # This prevents the template matcher from even looking at the spinner,
        # left edge, or scrollbar when anchors place the cursor elsewhere.
        if sorted_anchor_idxs:
            pos_in_sorted = bisect.bisect_right(sorted_anchor_idxs, i)
            prev_ai = sorted_anchor_idxs[pos_in_sorted - 1] if pos_in_sorted > 0 else None
            next_ai = sorted_anchor_idxs[pos_in_sorted] if pos_in_sorted < len(sorted_anchor_idxs) else None
            if prev_ai is not None and next_ai is not None:
                ax0, ay0 = anchor_map[prev_ai]
                ax1, ay1 = anchor_map[next_ai]
                margin = 220
                abbox = (
                    max(0, min(ax0, ax1) - margin),
                    max(0, min(ay0, ay1) - margin),
                    min(width, max(ax0, ax1) + margin),
                    min(height, max(ay0, ay1) + margin),
                )
                if search_region is not None:
                    intersected = (
                        max(search_region[0], abbox[0]),
                        max(search_region[1], abbox[1]),
                        min(search_region[2], abbox[2]),
                        min(search_region[3], abbox[3]),
                    )
                    # Only use intersection if it's large enough to search
                    if intersected[2] - intersected[0] >= 60 and intersected[3] - intersected[1] >= 60:
                        search_region = intersected
                    else:
                        search_region = abbox  # fall back to anchor box alone
                else:
                    search_region = abbox

        # 2. Template matching (runs every frame)
        tmatch = _detect_cursor_template(
            curr_gray, templates, search_region=search_region,
            risk_zones=risk_zones, prev_cursor_type=prev_cursor_type,
            y_min=y_min)

        # During busy frames, reject low-confidence template matches unless they
        # agree with the LK position (two signals agreeing = more trustworthy).
        if busy_frame and tmatch is not None and tmatch[2] < busy_min_conf:
            if lk_cx is not None:
                d = np.hypot(tmatch[0] - lk_cx, tmatch[1] - lk_cy)
                if d > 30 or tmatch[2] < 0.50:
                    tmatch = None  # disagrees with flow or too weak
            else:
                tmatch = None

        # LK agreement gate (all frames): when LK is confidently tracking the
        # cursor tip, any template match more than 80px away is almost certainly
        # a false positive (spinner, scrollbar, corner artifact).  Only override
        # LK when the template is extremely confident (>0.85 = near-perfect match).
        if tmatch is not None and lk_cx is not None and lk_conf >= 0.50:
            lk_dist = np.hypot(tmatch[0] - lk_cx, tmatch[1] - lk_cy)
            if lk_dist > 80 and tmatch[2] < 0.85:
                tmatch = None  # trust LK over disagreeing template

        cmatch = None
        if tmatch is not None:
            template_conf = tmatch[2]
            template_name = tmatch[3]
            cx, cy = tmatch[0], tmatch[1]
            # Re-anchor LK tracker at the confirmed template position
            lk_point = np.array([[[float(cx), float(cy)]]], dtype=np.float32)
        elif lk_cx is not None:
            # Template failed; use LK tracking result
            cx, cy = lk_cx, lk_cy

        # 2b. Color-based fallback (skip during busy frames — scrolling red
        # text/links cause false positives in color detection).
        # BGR is read on-demand from cap_color to avoid buffering all frames.
        is_scroll_frame = curr_idx in scroll_frame_set
        if cx is None and has_color_cursor and not busy_frame and not is_scroll_frame:
            cap_color.set(cv2.CAP_PROP_POS_FRAMES, curr_idx)
            ret_c, curr_bgr = cap_color.read()
            if ret_c and curr_bgr is not None:
                cmatch = _detect_cursor_color(
                    curr_bgr, search_region=search_region,
                    last_pos=last_pos, y_min=y_min)
                if cmatch is not None:
                    color_conf = cmatch[2]
                    cx, cy = cmatch[0], cmatch[1]

        # 3. Motion validation — only used as a CONFIDENCE BOOST for an
        # already-detected position, never as a standalone detector.
        # Motion-only detection (three-frame AND intersection) was removed:
        # spinners, scrollbars, and animated page elements all produce
        # cursor-sized motion blobs that caused too many false positives.
        if not busy_frame and cx is not None and i >= 1 and i < len(frame_buffer) - 1:
            prev_gray_m = frame_buffer[i - 1][1]
            next_gray_m = frame_buffer[i + 1][1]
            motion_conf = _motion_validate(
                prev_gray_m, curr_gray, next_gray_m, (cx, cy),
                y_min=y_min)

        # 4. Combine confidence — template > flow > motion hierarchy.
        # Require stronger motion confirmation (0.4) before granting the
        # template+motion bonus, to avoid false boosts from background motion.
        best_detect = max(template_conf, color_conf)
        if best_detect > 0 and motion_conf > 0.4:
            conf = min(0.95, 0.85 + motion_conf * 0.1)
            method = "template+motion"
        elif best_detect > 0 and lk_conf > 0:
            conf = min(0.92, best_detect + 0.08)
            method = "template+flow"
        elif best_detect > 0:
            conf = best_detect * 0.85
            method = "template" if template_conf >= color_conf else "color"
        elif lk_conf > 0 and cx is not None:
            conf = lk_conf
            method = "flow"
        elif motion_conf > 0 and cx is not None:
            conf = motion_conf * 0.6
            method = "motion"

        # 5a. Exclusion zone rejection — any detection inside a user-defined
        # exclusion zone is discarded entirely regardless of confidence.
        if cx is not None and exclusions:
            for ex, ey, er in exclusions:
                if np.hypot(cx - ex, cy - ey) <= er:
                    cx, cy = None, None
                    conf = 0.0
                    method = "hold"
                    lk_point = None  # don't propagate from an excluded spot
                    break

        # 5b. Jump rejection — reject detections that teleport too far from
        # last confident position unless confidence is very high or we've been
        # lost for a while (frames_since_detection > 10)
        if cx is not None and conf > 0 and last_confident_pos is not None:
            jump_dist = np.sqrt(
                (cx - last_confident_pos[0]) ** 2 +
                (cy - last_confident_pos[1]) ** 2)
            if jump_dist > 400 and frames_since_detection < 10:
                # Only accept if confidence is very high (template+motion)
                if conf < 0.80:
                    # Reject — treat as hold instead; also reset LK so it
                    # doesn't continue propagating the bad position
                    cx, cy = None, None
                    conf = 0.0
                    method = "hold"
                    lk_point = None

        # 6. Emit position
        if cx is not None and conf > 0:
            # Smooth moderate jumps (50-150px) by blending with Kalman prediction
            if last_pos is not None and kalman_pred is not None:
                det_dist = np.sqrt((cx - last_pos[0]) ** 2 + (cy - last_pos[1]) ** 2)
                if 50 < det_dist < 150 and conf < 0.85:
                    # Blend: weight toward detection based on confidence
                    blend = conf  # higher conf = trust detection more
                    cx = int(cx * blend + kalman_pred[0] * (1 - blend))
                    cy = int(cy * blend + kalman_pred[1] * (1 - blend))

            # Track if position is stuck (same spot = likely false match)
            if last_pos is not None:
                dist = np.sqrt((cx - last_pos[0]) ** 2 + (cy - last_pos[1]) ** 2)
                if dist < 5:
                    frames_at_same_pos += 1
                else:
                    frames_at_same_pos = 0
            else:
                frames_at_same_pos = 0

            last_pos = (cx, cy)
            if conf >= 0.5:
                last_confident_pos = (cx, cy)
            prev_cursor_type = template_name if template_name else prev_cursor_type
            frames_since_detection = 0
            detected += 1
            kalman.correct(cx, cy, conf)
            result.positions.append(CursorPosition(
                frame_idx=curr_idx, timestamp=round(timestamp, 3),
                x=cx, y=cy, confidence=round(conf, 3), method=method,
            ))
        elif last_pos is not None:
            # Hold: freeze at last CONFIDENT position.
            # Do NOT use Kalman velocity extrapolation — the accumulated
            # velocity vector can push the hold position toward screen edges
            # over several frames, causing "drift to corner" artifacts.
            frames_since_detection += 1
            frames_at_same_pos = 0  # not detecting = not stuck
            hx, hy = last_confident_pos if last_confident_pos is not None else last_pos
            hx = max(0, min(width - 1, hx))
            hy = max(0, min(height - 1, hy))
            result.positions.append(CursorPosition(
                frame_idx=curr_idx, timestamp=round(timestamp, 3),
                x=hx, y=hy,
                confidence=0.1, method="hold",
            ))

        if progress_callback and i % 100 == 0:
            progress_callback(curr_idx, total_frames)

    if cap_color is not None:
        cap_color.release()

    result.scrolls = scrolls
    result.frames_analyzed = len(frame_buffer)
    result.cursor_detected_count = detected

    # Second pass: anchor corridor filter, then preserve raw, then smooth
    if anchor_map:
        result.positions = _apply_anchor_corridor(result.positions, width, height)
    result.raw_positions = list(result.positions)
    result.positions = _smooth_positions(result.positions)
    return result


# ---------------------------------------------------------------------------
# Anchor corridor filter
# ---------------------------------------------------------------------------

def _apply_anchor_corridor(positions, width, height):
    """
    Between consecutive user-confirmed anchor positions, demote any automatic
    detection that falls outside a spatial corridor around the straight-line
    path between those two anchors.

    Uses perpendicular distance from the anchor-to-anchor segment (capsule
    geometry), NOT distance from the time-interpolated LERP point.  This
    allows natural curved cursor paths between anchors while still rejecting
    false positives that are spatially far from the true trajectory (spinner,
    scrollbar, left edge, etc.).

    Corridor radius is adaptive:
        max(150, min(500, anchor_dist * 1.2 + 100))
    so close anchors produce a tight corridor and distant ones allow more room.

    Demoted positions become method="interpolated" with coordinates linearly
    interpolated between the two bounding anchors.
    """
    if len(positions) < 2:
        return list(positions)

    anchor_indices = [(i, p) for i, p in enumerate(positions) if p.method == "anchor"]
    if len(anchor_indices) < 2:
        return list(positions)

    def _seg_dist(px, py, ax, ay, bx, by):
        dx, dy = bx - ax, by - ay
        seg_sq = dx * dx + dy * dy
        if seg_sq < 1:
            return np.hypot(px - ax, py - ay)
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_sq))
        return np.hypot(px - (ax + t * dx), py - (ay + t * dy))

    pts = list(positions)
    demoted = 0

    for span_idx in range(len(anchor_indices) - 1):
        i0, a0 = anchor_indices[span_idx]
        i1, a1 = anchor_indices[span_idx + 1]
        if i1 <= i0 + 1:
            continue

        anchor_dist = np.hypot(a1.x - a0.x, a1.y - a0.y)
        corridor_r = max(150, min(500, anchor_dist * 1.2 + 100))
        span_len = i1 - i0

        for k in range(i0 + 1, i1):
            p = pts[k]
            if p.method in ("anchor", "hold"):
                continue

            if _seg_dist(p.x, p.y, a0.x, a0.y, a1.x, a1.y) > corridor_r:
                t = (k - i0) / span_len
                ix = int(round(a0.x + (a1.x - a0.x) * t))
                iy = int(round(a0.y + (a1.y - a0.y) * t))
                interp_conf = round(min(a0.confidence, a1.confidence) * 0.75, 3)
                pts[k] = CursorPosition(
                    frame_idx=p.frame_idx, timestamp=p.timestamp,
                    x=ix, y=iy, confidence=interp_conf, method="interpolated",
                )
                demoted += 1

    if demoted:
        print(f"  Corridor filter: demoted {demoted} out-of-corridor detection(s) to interpolated")
    return pts


# ---------------------------------------------------------------------------
# Second-pass smoother
# ---------------------------------------------------------------------------

def _smooth_positions(positions, max_hold_gap=15, outlier_self_dist=100,
                      outlier_neighbor_dist=60, anchor_min_conf=0.65):
    """
    Retrospective second-pass smoother over first-pass CursorPosition list.

    Two operations applied in order:

    1. Outlier (jump-return) rejection — position i is a false positive when:
         dist(i, i-1) > outlier_self_dist  AND
         dist(i, i+1) > outlier_self_dist  AND
         dist(i-1, i+1) < outlier_neighbor_dist
       The classic jump-and-snap-back signature. Replaced by the midpoint of
       its two neighbors.  "template+motion" positions are never touched.

    2. Hold-gap interpolation — a contiguous run of "hold" positions that is
       ≤ max_hold_gap long AND bounded on both sides by confident anchors
       (confidence ≥ anchor_min_conf) is linearly interpolated between those
       anchors.  Longer gaps are left alone — the cursor was genuinely lost.

    Returns a new list of CursorPosition objects; the originals are untouched.
    """
    if len(positions) < 3:
        return list(positions)

    # Work on shallow copies so the raw list is never mutated
    pts = list(positions)

    def _copy(p, x, y, confidence, method):
        return CursorPosition(
            frame_idx=p.frame_idx, timestamp=p.timestamp,
            x=x, y=y, confidence=round(confidence, 3), method=method,
        )

    def _dist(a, b):
        return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

    # --- Pass 1: outlier (jump-return) rejection ---
    for i in range(1, len(pts) - 1):
        p, prev, nxt = pts[i], pts[i - 1], pts[i + 1]
        if p.method in ("template+motion", "anchor"):
            continue  # highest-confidence — never override
        if (_dist(p, prev) > outlier_self_dist and
                _dist(p, nxt) > outlier_self_dist and
                _dist(prev, nxt) < outlier_neighbor_dist):
            conf = min(prev.confidence, nxt.confidence) * 0.9
            pts[i] = _copy(p,
                           x=int(round((prev.x + nxt.x) / 2)),
                           y=int(round((prev.y + nxt.y) / 2)),
                           confidence=conf, method="interpolated")

    # --- Pass 2: hold-gap interpolation ---
    i = 0
    while i < len(pts):
        if pts[i].method != "hold":
            i += 1
            continue

        # Found the start of a hold run — find its extent
        run_start = i
        while i < len(pts) and pts[i].method == "hold":
            i += 1
        run_end = i - 1  # inclusive

        run_len = run_end - run_start + 1
        if run_len > max_hold_gap:
            continue  # too long — genuinely lost, don't fabricate a path

        # Need anchors on both sides
        if run_start == 0 or run_end == len(pts) - 1:
            continue

        a0 = pts[run_start - 1]
        a1 = pts[run_end + 1]
        if a0.confidence < anchor_min_conf or a1.confidence < anchor_min_conf:
            continue  # weak anchors — skip

        # Linearly interpolate x,y between the two anchors
        steps = run_len + 1  # total intervals from a0 to a1
        interp_conf = min(a0.confidence, a1.confidence) * 0.8
        for j, idx in enumerate(range(run_start, run_end + 1)):
            t = (j + 1) / steps  # 0 < t < 1
            pts[idx] = _copy(pts[idx],
                             x=int(round(a0.x + (a1.x - a0.x) * t)),
                             y=int(round(a0.y + (a1.y - a0.y) * t)),
                             confidence=interp_conf, method="interpolated")

    return pts


# ---------------------------------------------------------------------------
# Click detection
# ---------------------------------------------------------------------------

def _compute_ssim(region1, region2):
    """Compute SSIM between two same-sized grayscale regions."""
    r1 = region1.astype(np.float32)
    r2 = region2.astype(np.float32)
    mean1, mean2 = r1.mean(), r2.mean()
    std1, std2 = r1.std(), r2.std()
    if std1 < 1e-6 or std2 < 1e-6:
        return 1.0 if abs(mean1 - mean2) < 1.0 else 0.0
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    cov = ((r1 - mean1) * (r2 - mean2)).mean()
    ssim = ((2 * mean1 * mean2 + C1) * (2 * cov + C2)) / \
           ((mean1 ** 2 + mean2 ** 2 + C1) * (std1 ** 2 + std2 ** 2 + C2))
    return float(ssim)


def _local_ssim(frame1, frame2, cx, cy, radius=120):
    """SSIM in a local region around (cx, cy)."""
    h, w = frame1.shape[:2]
    x1, y1 = max(0, cx - radius), max(0, cy - radius)
    x2, y2 = min(w, cx + radius), min(h, cy + radius)
    r1 = frame1[y1:y2, x1:x2]
    r2 = frame2[y1:y2, x1:x2]
    if r1.size == 0 or r2.size == 0:
        return 1.0
    return _compute_ssim(r1, r2)


def _global_ssim(frame1, frame2):
    """Global SSIM between two frames (downsampled)."""
    f1 = cv2.resize(frame1, (640, 360))
    f2 = cv2.resize(frame2, (640, 360))
    return _compute_ssim(f1, f2)


def detect_clicks(video_path, tracking_result):
    """Detect clicks via visual changes near the cursor, excluding scroll frames."""
    if len(tracking_result.positions) < 3:
        return

    # Build set of scroll frame indices for exclusion
    scroll_frames = set()
    for s in tracking_result.scrolls:
        # Mark a window around each scroll event
        for offset in range(-3, 4):
            scroll_frames.add(s.frame_idx + offset * tracking_result.sample_step)

    cap = cv2.VideoCapture(video_path)
    pos_map = {p.frame_idx: p for p in tracking_result.positions}
    frame_indices = sorted(pos_map.keys())

    # Pre-compute velocities for pause detection
    velocities = {}
    for i in range(1, len(tracking_result.positions)):
        p0 = tracking_result.positions[i - 1]
        p1 = tracking_result.positions[i]
        dt = p1.timestamp - p0.timestamp
        v = np.sqrt((p1.x - p0.x) ** 2 + (p1.y - p0.y) ** 2) / dt if dt > 0 else 0
        velocities[p1.frame_idx] = v

    # (pause_frames signal removed — too many false positives from hovering)

    clicks = []
    prev_gray = None
    prev_fidx = None

    for fi_idx, fidx in enumerate(frame_indices):
        # Skip scroll frames
        if fidx in scroll_frames:
            prev_fidx = fidx
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, frame = cap.read()
            if ret:
                prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None and prev_fidx is not None:
            pos = pos_map[fidx]
            # Only consider click if cursor was actually detected (not just held)
            if pos.method == "hold":
                prev_gray = gray
                prev_fidx = fidx
                continue

            local = _local_ssim(prev_gray, gray, pos.x, pos.y)
            glob = _global_ssim(prev_gray, gray)

            is_click = False
            kind = ""
            conf = 0.0

            # Signal 1: local visual change + cursor near-stationary
            # Threshold 0.65 (was 0.70) — stricter to reduce false positives
            if local < 0.65:
                prev_pos = pos_map.get(prev_fidx)
                if prev_pos:
                    dist = np.sqrt((pos.x - prev_pos.x) ** 2 + (pos.y - prev_pos.y) ** 2)
                    if dist < 60:
                        is_click = True
                        kind = "visual_change"
                        conf = max(0, 1.0 - local)

            # Signal 2: scene change (implies a preceding click/navigation)
            if glob < 0.6:
                sc = min(1.0, max(0, 1.0 - glob))
                if sc > conf:
                    is_click = True
                    kind = "scene_change"
                    conf = sc

            # Note: pause_click signal removed — velocity pauses happen during
            # hovering, reading, and menu browsing, not just clicks. The visual
            # change signal is a more reliable indicator.

            if is_click and conf > 0.35:
                clicks.append(ClickEvent(
                    frame_idx=fidx, timestamp=round(pos.timestamp, 3),
                    x=pos.x, y=pos.y,
                    confidence=round(conf, 3), kind=kind,
                ))

        prev_gray = gray
        prev_fidx = fidx

    cap.release()

    # Debounce: merge within 0.5s
    if clicks:
        merged = [clicks[0]]
        for c in clicks[1:]:
            if c.timestamp - merged[-1].timestamp > 0.5:
                merged.append(c)
            elif c.confidence > merged[-1].confidence:
                merged[-1] = c
        clicks = merged

    tracking_result.clicks = clicks


# ---------------------------------------------------------------------------
# Scene detection
# ---------------------------------------------------------------------------

def detect_scenes(video_path, tracking_result, ssim_threshold=0.65, min_scene_duration=0.5):
    """Detect scene/page changes using global frame SSIM."""
    cap = cv2.VideoCapture(video_path)
    fps = tracking_result.fps
    total_frames = tracking_result.total_frames
    scene_sample = max(1, int(fps / 4))

    prev_gray_small = None
    scene_boundaries = [0]
    screenshots = {}

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % scene_sample == 0:
            small = cv2.resize(frame, (640, 360))
            gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if prev_gray_small is not None:
                ssim = _compute_ssim(prev_gray_small, gray_small)
                if ssim < ssim_threshold:
                    scene_boundaries.append(frame_idx)
                    screenshots[frame_idx] = frame.copy()
            if frame_idx == 0:
                screenshots[0] = frame.copy()
            prev_gray_small = gray_small
        frame_idx += 1
    cap.release()

    scene_boundaries.append(total_frames)

    # Debounce
    min_frames = int(min_scene_duration * fps)
    debounced = [scene_boundaries[0]]
    for b in scene_boundaries[1:]:
        if b - debounced[-1] >= min_frames:
            debounced.append(b)
    if debounced[-1] != total_frames:
        debounced.append(total_frames)
    scene_boundaries = debounced

    # Build scenes
    scene_screenshots = []
    scenes = []
    for i in range(len(scene_boundaries) - 1):
        start = scene_boundaries[i]
        end = scene_boundaries[i + 1]
        best_key = min(screenshots.keys(), key=lambda k: abs(k - start)) if screenshots else 0
        scene_screenshots.append(screenshots.get(best_key))

        click_count = sum(1 for c in tracking_result.clicks if start <= c.frame_idx < end)
        scene_scrolls = [s for s in tracking_result.scrolls if start <= s.frame_idx < end]
        scroll_total = sum(s.pixels for s in scene_scrolls)
        scroll_time = _scroll_time_s(scene_scrolls, tracking_result.sample_step, fps)

        scenes.append(Scene(
            start_frame=start, end_frame=end,
            start_time=round(start / fps, 2), end_time=round(end / fps, 2),
            screenshot_idx=len(scene_screenshots) - 1,
            click_count=click_count, scroll_total=scroll_total, scroll_time_s=scroll_time,
        ))

    # Backtrack detection
    for i in range(len(scenes)):
        if scene_screenshots[i] is None:
            continue
        curr_hist = cv2.calcHist(
            [cv2.cvtColor(scene_screenshots[i], cv2.COLOR_BGR2GRAY)],
            [0], None, [64], [0, 256])
        curr_hist = cv2.normalize(curr_hist, curr_hist).flatten()
        for j in range(i - 1, max(-1, i - 10), -1):
            if scene_screenshots[j] is None:
                continue
            prev_hist = cv2.calcHist(
                [cv2.cvtColor(scene_screenshots[j], cv2.COLOR_BGR2GRAY)],
                [0], None, [64], [0, 256])
            prev_hist = cv2.normalize(prev_hist, prev_hist).flatten()
            if cv2.compareHist(curr_hist, prev_hist, cv2.HISTCMP_CORREL) > 0.95:
                scenes[i].is_backtrack = True
                scenes[i].similar_scene_idx = j
                break

    tracking_result.scenes = scenes

    # Extract URL bar crops for scene labeling
    url_bar_crops = []
    url_region = getattr(tracking_result, 'url_bar_region', None)
    if url_region:
        uy1, uy2 = url_region
        for ss in scene_screenshots:
            if ss is not None:
                url_bar_crops.append(ss[uy1:uy2, 60:ss.shape[1] // 2, :].copy())
            else:
                url_bar_crops.append(None)
    else:
        url_bar_crops = [None] * len(scene_screenshots)

    return scene_screenshots, url_bar_crops


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def generate_heatmap(tracking_result, output_path):
    """Generate heatmap on a dark background (page-independent screen position map)."""
    width, height = tracking_result.width, tracking_result.height
    scale = 0.5
    acc_w, acc_h = int(width * scale), int(height * scale)
    accumulator = np.zeros((acc_h, acc_w), dtype=np.float64)

    for pos in tracking_result.positions:
        if pos.confidence < 0.2:
            continue  # skip low-confidence holds
        sx, sy = int(pos.x * scale), int(pos.y * scale)
        if 0 <= sx < acc_w and 0 <= sy < acc_h:
            accumulator[sy, sx] += 1.0

    ksize = int(121 * scale) | 1
    accumulator = cv2.GaussianBlur(accumulator, (ksize, ksize), 0)
    if accumulator.max() > 0:
        accumulator /= accumulator.max()
    accumulator = np.power(accumulator, 0.35)
    if accumulator.max() > 0:
        accumulator /= accumulator.max()

    heatmap = cv2.applyColorMap((accumulator * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (width, height))

    # Dark background instead of first frame — positions span multiple pages
    bg = np.full((height, width, 3), 30, dtype=np.uint8)
    mask = cv2.resize(accumulator, (width, height))
    mask[mask < 0.05] = 0
    mask = np.clip(mask * 1.5, 0, 0.9)
    m3 = np.stack([mask] * 3, axis=-1)
    blended = (bg.astype(np.float64) * (1 - m3) + heatmap.astype(np.float64) * m3).astype(np.uint8)

    # Add axis labels
    cv2.putText(blended, "Screen Position Heatmap (all pages combined)",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)

    cv2.imwrite(output_path, blended)


def _time_color(t):
    """Map t in [0,1] to BGR color: blue -> cyan -> green -> yellow -> red."""
    t = max(0.0, min(1.0, t))
    if t < 0.25:
        r, g, b = 0, int(t * 4 * 255), 255
    elif t < 0.5:
        r, g, b = 0, 255, int((0.5 - t) * 4 * 255)
    elif t < 0.75:
        r, g, b = int((t - 0.5) * 4 * 255), 255, 0
    else:
        r, g, b = 255, int((1.0 - t) * 4 * 255), 0
    return (b, g, r)


def _draw_clicks_on(img, clicks):
    """Draw click markers on an image."""
    colors = {
        "visual_change": (0, 255, 255),
        "scene_change": (0, 0, 255),
        "pause_click": (255, 165, 0),
    }
    for click in clicks:
        r = int(20 + click.confidence * 30)
        c = colors.get(click.kind, (0, 255, 0))
        cv2.circle(img, (click.x, click.y), r, c, 3)
        cv2.circle(img, (click.x, click.y), 6, c, -1)
        cv2.putText(img, f"{click.kind} ({click.confidence:.1f})",
                    (click.x + r + 5, click.y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
    return img


def _draw_trajectory_on(img, positions, darken=True):
    """Draw trajectory lines on an image."""
    if darken:
        img[:] = (img.astype(np.float64) * 0.3).astype(np.uint8)
    if len(positions) < 2:
        return img
    t0, t1 = positions[0].timestamp, positions[-1].timestamp
    total = max(t1 - t0, 0.001)
    for i in range(1, len(positions)):
        p0, p1 = positions[i - 1], positions[i]
        t = (p1.timestamp - t0) / total
        color = _time_color(t)
        thick = 1 if p1.method == "hold" else 2
        cv2.line(img, (p0.x, p0.y), (p1.x, p1.y), color, thick)
    cv2.circle(img, (positions[0].x, positions[0].y), 15, (0, 255, 0), -1)
    cv2.putText(img, "START", (positions[0].x + 20, positions[0].y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.circle(img, (positions[-1].x, positions[-1].y), 15, (0, 0, 255), -1)
    cv2.putText(img, "END", (positions[-1].x + 20, positions[-1].y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return img


def generate_per_scene_visuals(tracking_result, scene_screenshots, url_bar_crops, output_dir):
    """Generate per-scene click maps, trajectory images, and URL bar crops."""
    scenes_dir = os.path.join(output_dir, "scenes")
    os.makedirs(scenes_dir, exist_ok=True)
    scene_visuals = []

    for i, scene in enumerate(tracking_result.scenes):
        vis = {"click_map": None, "trajectory": None, "url_bar": None}

        # Save URL bar crop
        if i < len(url_bar_crops) and url_bar_crops[i] is not None:
            url_path = os.path.join(scenes_dir, f"scene_{i+1:02d}_url.png")
            cv2.imwrite(url_path, url_bar_crops[i])
            vis["url_bar"] = url_path
        bg = scene_screenshots[i].copy() if i < len(scene_screenshots) and scene_screenshots[i] is not None \
            else np.zeros((tracking_result.height, tracking_result.width, 3), dtype=np.uint8)

        scene_clicks = [c for c in tracking_result.clicks if scene.start_frame <= c.frame_idx < scene.end_frame]
        scene_positions = [p for p in tracking_result.positions if scene.start_frame <= p.frame_idx < scene.end_frame]
        scene_scrolls = [s for s in tracking_result.scrolls if scene.start_frame <= s.frame_idx < scene.end_frame]

        dur = scene.end_time - scene.start_time
        scroll_info = f"  |  scroll: {sum(s.pixels for s in scene_scrolls)}px" if scene_scrolls else ""
        label = f"Scene {i+1}  |  {scene.start_time:.1f}s - {scene.end_time:.1f}s  |  {len(scene_clicks)} clicks{scroll_info}"

        # Click map
        click_img = bg.copy()
        if scene_clicks:
            _draw_clicks_on(click_img, scene_clicks)
        else:
            cv2.putText(click_img, "No clicks detected", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 128, 128), 2)
        cv2.rectangle(click_img, (0, 0), (min(len(label) * 14, click_img.shape[1]), 40), (0, 0, 0), -1)
        cv2.putText(click_img, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        click_path = os.path.join(scenes_dir, f"scene_{i+1:02d}_clicks.png")
        cv2.imwrite(click_path, click_img)
        vis["click_map"] = click_path

        # Trajectory
        traj_img = bg.copy()
        if scene_positions:
            _draw_trajectory_on(traj_img, scene_positions, darken=True)
        else:
            traj_img[:] = (traj_img.astype(np.float64) * 0.3).astype(np.uint8)
            cv2.putText(traj_img, "No cursor data", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 128, 128), 2)
        cv2.rectangle(traj_img, (0, 0), (min(len(label) * 14, traj_img.shape[1]), 40), (0, 0, 0), -1)
        cv2.putText(traj_img, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        traj_path = os.path.join(scenes_dir, f"scene_{i+1:02d}_trajectory.png")
        cv2.imwrite(traj_path, traj_img)
        vis["trajectory"] = traj_path

        scene_visuals.append(vis)
    return scene_visuals


def generate_scene_gifs(video_path, tracking_result, scene_visuals, output_dir,
                        max_frames_per_gif=60, gif_scale=0.35, gif_fps=8):
    """Generate an animated GIF for each scene showing the actual video with cursor overlay."""
    scenes = tracking_result.scenes
    if not scenes:
        return

    scenes_dir = os.path.join(output_dir, "scenes")
    os.makedirs(scenes_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    video_fps = tracking_result.fps

    # Build position and click lookups for overlay
    pos_map = {p.frame_idx: p for p in tracking_result.positions}
    # Click lookup: frame_idx → ClickEvent (within a small window for GIF frames)
    click_map = {}
    for c in tracking_result.clicks:
        click_map[c.frame_idx] = c

    for i, scene in enumerate(scenes):
        start_f = scene.start_frame
        end_f = scene.end_frame
        scene_frames = end_f - start_f

        # Sample evenly to stay within max_frames_per_gif
        step = max(1, scene_frames // max_frames_per_gif)
        frames_rgb = []

        for fidx in range(start_f, end_f, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Resize for GIF
            h, w = frame.shape[:2]
            new_w, new_h = int(w * gif_scale), int(h * gif_scale)
            small = cv2.resize(frame, (new_w, new_h))

            # Draw cursor position if available
            # Find nearest tracked position
            nearest_pos = None
            for offset in range(0, step + 1):
                if fidx + offset in pos_map:
                    nearest_pos = pos_map[fidx + offset]
                    break
                if fidx - offset in pos_map:
                    nearest_pos = pos_map[fidx - offset]
                    break

            if nearest_pos is not None:
                gx = int(nearest_pos.x * gif_scale)
                gy = int(nearest_pos.y * gif_scale)
                m = nearest_pos.method
                # Color-code by detection method for easy debugging
                if m == "anchor":
                    color = (0, 215, 255)  # gold    — user-confirmed anchor
                elif m == "template+motion":
                    color = (0, 255, 0)    # green   — template + motion (highest)
                elif m == "template+flow":
                    color = (255, 80, 0)   # blue    — template re-anchored by LK
                elif m in ("template", "color"):
                    color = (0, 220, 255)  # yellow  — template only
                elif m == "flow":
                    color = (200, 0, 200)  # magenta — LK optical flow only
                elif m == "interpolated":
                    color = (0, 165, 255)  # orange  — second-pass interpolated
                elif m == "motion":
                    color = (0, 255, 255)  # yellow  — motion only
                else:
                    color = (140, 140, 140)  # gray — hold / extrapolated
                r = 12
                cv2.circle(small, (gx, gy), r, color, 2)
                cv2.line(small, (gx - r - 4, gy), (gx + r + 4, gy), color, 1)
                cv2.line(small, (gx, gy - r - 4), (gx, gy + r + 4), color, 1)
                # Short method label next to marker
                label = m[:4] if m else "?"
                cv2.putText(small, label, (gx + r + 2, gy + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)

            # Draw click marker if a click was detected near this frame
            nearby_click = None
            for off in range(0, step + 1):
                if fidx + off in click_map:
                    nearby_click = click_map[fidx + off]
                    break
                if off > 0 and fidx - off in click_map:
                    nearby_click = click_map[fidx - off]
                    break
            if nearby_click is not None:
                kx = int(nearby_click.x * gif_scale)
                ky = int(nearby_click.y * gif_scale)
                # Red pulsing ring to indicate detected click
                cv2.circle(small, (kx, ky), 18, (0, 0, 255), 2)
                cv2.circle(small, (kx, ky), 6, (0, 0, 255), -1)
                cv2.putText(small, nearby_click.kind[:3], (kx + 10, ky - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1, cv2.LINE_AA)

            # Convert BGR to RGB for PIL
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            frames_rgb.append(rgb)

            if len(frames_rgb) >= max_frames_per_gif:
                break

        if not frames_rgb:
            continue

        # Write GIF using PIL
        from PIL import Image
        pil_frames = [Image.fromarray(f) for f in frames_rgb]
        gif_path = os.path.join(scenes_dir, f"scene_{i+1:02d}.gif")
        frame_duration = max(50, int(1000 / gif_fps))  # ms per frame
        pil_frames[0].save(
            gif_path, save_all=True, append_images=pil_frames[1:],
            duration=frame_duration, loop=0, optimize=True,
        )

        # Add to visuals
        if i < len(scene_visuals):
            scene_visuals[i]["gif"] = gif_path

    cap.release()


def generate_journey_map(tracking_result, scene_screenshots, output_path):
    """Generate timeline visualization."""
    scenes = tracking_result.scenes
    if not scenes:
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.text(0.5, 0.5, "No scenes detected", ha="center", va="center", fontsize=14)
        ax.axis("off")
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#1e1e1e")
        plt.close(fig)
        return

    n = len(scenes)
    fig_w = max(12, n * 2.5)
    fig = plt.figure(figsize=(fig_w, 6), facecolor="#1e1e1e")

    ax = fig.add_axes([0.05, 0.1, 0.9, 0.25])
    ax.set_facecolor("#2d2d2d")
    total_dur = tracking_result.duration
    colors_list = plt.cm.Set3(np.linspace(0, 1, max(n, 1)))

    for i, sc in enumerate(scenes):
        dur = sc.end_time - sc.start_time
        ax.barh(0, dur, left=sc.start_time, height=0.8,
                color=colors_list[i % len(colors_list)], edgecolor="white", linewidth=0.5)
        if dur > total_dur * 0.05:
            mid = sc.start_time + dur / 2
            label = f"Scene {i+1}\n{dur:.1f}s"
            if sc.click_count > 0:
                label += f"\n{sc.click_count} clicks"
            if sc.scroll_total > 0:
                label += f"\nscroll {sc.scroll_total}px"
            if sc.is_backtrack:
                label += "\n(backtrack)"
            ax.text(mid, 0, label, ha="center", va="center", fontsize=7, color="black", fontweight="bold")

    ax.set_xlim(0, total_dur)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel("Time (seconds)", color="white", fontsize=10)
    ax.tick_params(colors="white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])

    for i, sc in enumerate(scenes):
        if i >= len(scene_screenshots) or scene_screenshots[i] is None:
            continue
        thumb = cv2.cvtColor(cv2.resize(scene_screenshots[i], (200, 112)), cv2.COLOR_BGR2RGB)
        mid = (sc.start_time + sc.end_time) / 2
        x_frac = 0.05 + 0.9 * (mid / total_dur) - 0.075
        ax_t = fig.add_axes([max(0.02, min(0.95, x_frac)), 0.45, min(0.15, 0.9 / max(n, 1)), 0.45])
        ax_t.imshow(thumb)
        ax_t.axis("off")
        border = "red" if sc.is_backtrack else "white"
        for sp in ax_t.spines.values():
            sp.set_edgecolor(border)
            sp.set_linewidth(2)
            sp.set_visible(True)
        ax_t.set_title(f"Scene {i+1}", color="white", fontsize=8, pad=2)

    fig.suptitle("User Journey Map", color="white", fontsize=14, y=0.98)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#1e1e1e")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _img_b64(path):
    """Image file to base64 data URI."""
    if not path or not os.path.exists(path):
        return ""
    ext = os.path.splitext(path)[1].lower()
    mime = "image/gif" if ext == ".gif" else "image/png"
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{data}"


def generate_report(tracking_result, output_dir, scene_screenshots, scene_visuals):
    """Generate standalone HTML report."""
    heatmap_b64 = _img_b64(os.path.join(output_dir, "heatmap.png"))
    journey_b64 = _img_b64(os.path.join(output_dir, "journey.png"))

    r = tracking_result
    duration = r.duration
    n_clicks = len(r.clicks)
    n_scenes = len(r.scenes)
    detect_pct = r.cursor_detected_count / max(r.frames_analyzed, 1) * 100
    backtracks = sum(1 for s in r.scenes if s.is_backtrack)
    total_scroll = sum(s.pixels for s in r.scrolls)
    n_scroll_events = len(r.scrolls)
    total_scroll_time = _scroll_time_s(r.scrolls, r.sample_step, r.fps)

    total_distance = 0
    for i in range(1, len(r.positions)):
        p0, p1 = r.positions[i - 1], r.positions[i]
        total_distance += np.sqrt((p1.x - p0.x) ** 2 + (p1.y - p0.y) ** 2)
    avg_speed = total_distance / duration if duration > 0 else 0

    # Scene table
    scene_rows = ""
    for i, sc in enumerate(r.scenes):
        dur = sc.end_time - sc.start_time
        notes = ""
        if sc.is_backtrack:
            notes += '<span style="color:#f44336">backtrack</span> '
        if sc.scroll_total > 0:
            notes += f'scroll {sc.scroll_total}px ({sc.scroll_time_s:.1f}s)'
        scene_rows += f"<tr><td>{i+1}</td><td>{sc.start_time:.1f}s - {sc.end_time:.1f}s</td>" \
                      f"<td>{dur:.1f}s</td><td>{sc.click_count}</td><td>{notes}</td></tr>"

    # Click table
    click_rows = ""
    for i, cl in enumerate(r.clicks):
        sn = "?"
        for si, sc in enumerate(r.scenes):
            if sc.start_frame <= cl.frame_idx < sc.end_frame:
                sn = si + 1
                break
        click_rows += f"<tr><td>{i+1}</td><td>{cl.timestamp:.2f}s</td><td>({cl.x}, {cl.y})</td>" \
                      f"<td>{cl.kind}</td><td>{cl.confidence:.2f}</td><td>Scene {sn}</td></tr>"

    # Per-scene blocks
    per_scene_html = ""
    for i, sc in enumerate(r.scenes):
        dur = sc.end_time - sc.start_time
        bt = ' <span style="color:#f44336;font-weight:bold">[BACKTRACK]</span>' if sc.is_backtrack else ""
        scroll_note = f" | scroll {sc.scroll_total}px ({sc.scroll_time_s:.1f}s)" if sc.scroll_total > 0 else ""
        sv = scene_visuals[i] if i < len(scene_visuals) else {}
        cb64 = _img_b64(sv.get("click_map", ""))
        tb64 = _img_b64(sv.get("trajectory", ""))
        gb64 = _img_b64(sv.get("gif", ""))
        ub64 = _img_b64(sv.get("url_bar", ""))
        url_html = f'<img src="{ub64}" class="url-bar">' if ub64 else ""
        gif_html = f'<div class="scene-gif"><h4>Scene Recording</h4><img src="{gb64}"></div>' if gb64 else ""
        per_scene_html += f'''
        <div class="scene-block">
            <h3>Scene {i+1}{bt}
                <span class="scene-meta">{sc.start_time:.1f}s &ndash; {sc.end_time:.1f}s
                &nbsp;|&nbsp; {dur:.1f}s &nbsp;|&nbsp; {sc.click_count} clicks{scroll_note}</span>
            </h3>
            {url_html}
            {gif_html}
            <div class="scene-pair">
                <div class="scene-img"><h4>Click Map</h4><img src="{cb64}"></div>
                <div class="scene-img"><h4>Trajectory</h4><img src="{tb64}"></div>
            </div>
        </div>'''

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Mouse Tracking Report</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0f0f0f;color:#e0e0e0;padding:20px;max-width:1400px;margin:0 auto}}
h1{{color:#fff;margin-bottom:10px;font-size:28px}}
h2{{color:#9ecfff;margin:30px 0 15px;font-size:20px;border-bottom:1px solid #333;padding-bottom:8px}}
h3{{color:#e0e0e0;margin:20px 0 10px;font-size:16px}}
h4{{color:#888;font-size:13px;margin-bottom:6px;font-weight:500}}
.subtitle{{color:#888;margin-bottom:30px;font-size:14px}}
.scene-meta{{color:#888;font-size:13px;font-weight:normal}}
.stats-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:15px;margin-bottom:30px}}
.stat-card{{background:#1a1a2e;border:1px solid #333;border-radius:10px;padding:20px;text-align:center}}
.stat-value{{font-size:32px;font-weight:bold;color:#4fc3f7}}
.stat-label{{font-size:12px;color:#888;margin-top:5px;text-transform:uppercase}}
.viz-section{{margin-bottom:40px}}
.viz-section img{{width:100%;border-radius:8px;border:1px solid #333}}
.scene-block{{background:#141420;border:1px solid #2a2a3a;border-radius:10px;padding:15px 20px;margin-bottom:25px}}
.url-bar{{max-width:500px;border-radius:4px;border:1px solid #444;margin:6px 0 12px 0}}
.scene-gif{{margin-bottom:15px}}
.scene-gif img{{max-width:500px;border-radius:6px;border:1px solid #333}}
.scene-pair{{display:grid;grid-template-columns:1fr 1fr;gap:15px}}
.scene-img img{{width:100%;border-radius:6px;border:1px solid #333}}
table{{width:100%;border-collapse:collapse;margin-top:10px;font-size:14px}}
th,td{{padding:10px 15px;text-align:left;border-bottom:1px solid #222}}
th{{color:#9ecfff;font-weight:600;background:#1a1a2e}}
tr:hover{{background:#1a1a2e}}
</style>
</head>
<body>
<h1>Mouse Tracking Report</h1>
<p class="subtitle">
    Video: {os.path.basename(r.video_path)}<br>
    Resolution: {r.width}x{r.height} | FPS: {r.fps:.0f} |
    Duration: {duration:.1f}s | Frames analyzed: {r.frames_analyzed}
</p>

<h2>Summary Statistics</h2>
<div class="stats-grid">
    <div class="stat-card"><div class="stat-value">{n_clicks}</div><div class="stat-label">Detected Clicks</div></div>
    <div class="stat-card"><div class="stat-value">{n_scenes}</div><div class="stat-label">Scenes / Pages</div></div>
    <div class="stat-card"><div class="stat-value">{backtracks}</div><div class="stat-label">Backtracks</div></div>
    <div class="stat-card"><div class="stat-value">{detect_pct:.0f}%</div><div class="stat-label">Cursor Detection Rate</div></div>
    <div class="stat-card"><div class="stat-value">{total_distance:.0f}px</div><div class="stat-label">Total Mouse Distance</div></div>
    <div class="stat-card"><div class="stat-value">{avg_speed:.0f}px/s</div><div class="stat-label">Avg Mouse Speed</div></div>
    <div class="stat-card"><div class="stat-value">{n_scroll_events}</div><div class="stat-label">Scroll Events</div></div>
    <div class="stat-card"><div class="stat-value">{total_scroll}px</div><div class="stat-label">Total Scroll Distance</div></div>
    <div class="stat-card"><div class="stat-value">{total_scroll_time:.1f}s</div><div class="stat-label">Time Scrolling</div></div>
</div>

<h2>Screen Position Heatmap (all pages)</h2>
<div class="viz-section"><img src="{heatmap_b64}" alt="Heatmap"></div>

<h2>User Journey</h2>
<div class="viz-section"><img src="{journey_b64}" alt="Journey"></div>

<h2>Per-Scene Click Maps &amp; Trajectories</h2>
{per_scene_html}

<h2>Scene Details</h2>
<table>
<thead><tr><th>#</th><th>Time Range</th><th>Duration</th><th>Clicks</th><th>Notes</th></tr></thead>
<tbody>{scene_rows}</tbody>
</table>

<h2>Click Details</h2>
<table>
<thead><tr><th>#</th><th>Time</th><th>Position</th><th>Type</th><th>Confidence</th><th>Scene</th></tr></thead>
<tbody>{click_rows}</tbody>
</table>

<p style="margin-top:40px;color:#555;font-size:12px;text-align:center">Generated by Video Mouse Tracker</p>
</body></html>"""

    report_path = os.path.join(output_dir, "report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    return report_path


def save_raw_data(tracking_result, output_path):
    """Save tracking data as JSON."""
    data = {
        "video": {
            "path": tracking_result.video_path,
            "width": tracking_result.width, "height": tracking_result.height,
            "fps": tracking_result.fps, "total_frames": tracking_result.total_frames,
            "duration": tracking_result.duration,
        },
        "analysis": {
            "sample_step": tracking_result.sample_step,
            "frames_analyzed": tracking_result.frames_analyzed,
            "cursor_detected_count": tracking_result.cursor_detected_count,
            "detection_rate": round(tracking_result.cursor_detected_count / max(tracking_result.frames_analyzed, 1), 3),
        },
        "positions": [asdict(p) for p in tracking_result.positions],
        "raw_positions": [asdict(p) for p in tracking_result.raw_positions],
        "clicks": [asdict(c) for c in tracking_result.clicks],
        "scrolls": [asdict(s) for s in tracking_result.scrolls],
        "scenes": [asdict(s) for s in tracking_result.scenes],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python tracker.py <video_file.mp4>")
        sys.exit(1)

    video_path = sys.argv[1]
    if not os.path.isfile(video_path):
        print(f"Error: File not found: {video_path}")
        sys.exit(1)

    video_dir = os.path.dirname(os.path.abspath(video_path))
    video_name = Path(video_path).stem
    output_dir = os.path.join(video_dir, f"output_{video_name[:50]}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Video Mouse Tracker")
    print(f"{'=' * 60}")
    print(f"Input:  {video_path}")
    print(f"Output: {output_dir}")
    print()

    sample_step = 2

    # Look for a Cursors/ folder one level up from the Task folder
    # Expected layout: Session_Recordings/SessionN_Name/TaskN/video.mp4
    #                  Session_Recordings/SessionN_Name/Cursors/
    cursors_dir = Path(video_path).parent.parent / "Cursors"
    if not cursors_dir.is_dir():
        cursors_dir = None

    # Stage 1+2: Cursor tracking
    print("[1/7] Tracking cursor positions...")
    if cursors_dir:
        print(f"       Using custom cursors from: {cursors_dir}")

    # Collect user-confirmed anchor frames before tracking starts.
    # If a cached anchors.json exists next to the video it loads instantly;
    # otherwise an OpenCV window opens for interactive annotation.
    # Scale n_samples to video duration: ~1 anchor per 12 seconds, 16-30 range.
    _cap_meta = cv2.VideoCapture(video_path)
    _dur = int(_cap_meta.get(cv2.CAP_PROP_FRAME_COUNT)) / max(1.0, _cap_meta.get(cv2.CAP_PROP_FPS))
    _cap_meta.release()
    n_samples = max(16, min(30, int(_dur / 12)))
    print(f"       Collecting {n_samples} anchor points (~1 per 12s for {_dur:.0f}s video)...")
    anchors, exclusions = _collect_anchors(video_path, n_samples=n_samples)

    t0 = time.time()

    def progress(frame, total):
        print(f"       {frame/max(total,1)*100:.0f}% ({frame}/{total} frames)", end="\r")

    result = track_cursor(video_path, sample_step=sample_step, progress_callback=progress,
                          cursors_dir=str(cursors_dir) if cursors_dir else None,
                          anchors=anchors if anchors else None,
                          exclusions=exclusions if exclusions else None)
    n_scroll = len(result.scrolls)
    print(f"\n       Done. {result.cursor_detected_count}/{result.frames_analyzed} frames with cursor "
          f"({result.cursor_detected_count/max(result.frames_analyzed,1)*100:.0f}%), "
          f"{n_scroll} scroll events in {time.time()-t0:.1f}s")

    # Stage 3: Clicks
    print("[2/7] Detecting clicks...")
    t0 = time.time()
    detect_clicks(video_path, result)
    print(f"       Found {len(result.clicks)} clicks in {time.time()-t0:.1f}s")

    # Stage 4: Scenes
    print("[3/7] Detecting scenes/pages...")
    t0 = time.time()
    scene_screenshots, url_bar_crops = detect_scenes(video_path, result)
    bt = sum(1 for s in result.scenes if s.is_backtrack)
    print(f"       Found {len(result.scenes)} scenes ({bt} backtracks) in {time.time()-t0:.1f}s")

    # Stage 5: Visuals
    print("[4/7] Generating heatmap...")
    generate_heatmap(result, os.path.join(output_dir, "heatmap.png"))

    print("[5/7] Generating per-scene click maps & trajectories...")
    scene_visuals = generate_per_scene_visuals(result, scene_screenshots, url_bar_crops, output_dir)
    generate_journey_map(result, scene_screenshots, os.path.join(output_dir, "journey.png"))
    print(f"       Generated visuals for {len(scene_visuals)} scenes")

    print("[6/7] Generating per-scene GIFs...")
    t0 = time.time()
    generate_scene_gifs(video_path, result, scene_visuals, output_dir)
    print(f"       Generated GIFs in {time.time()-t0:.1f}s")

    # Stage 7: Report
    print("[7/7] Generating report...")
    save_raw_data(result, os.path.join(output_dir, "raw_data.json"))
    report_path = generate_report(result, output_dir, scene_screenshots, scene_visuals)

    print()
    print(f"{'=' * 60}")
    print(f"Complete! Output files:")
    for f in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, f)
        if os.path.isfile(fpath):
            print(f"  {f:30s} ({os.path.getsize(fpath)/1024:.0f} KB)")
        else:
            n_files = len(os.listdir(fpath))
            print(f"  {f + '/':30s} ({n_files} files)")
    print()
    print(f"Open report: {report_path}")


if __name__ == "__main__":
    main()
