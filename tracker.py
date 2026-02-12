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
    method: str  # "three_frame", "two_frame", "hold"


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
    positions: list = field(default_factory=list)
    clicks: list = field(default_factory=list)
    scrolls: list = field(default_factory=list)
    scenes: list = field(default_factory=list)
    frames_analyzed: int = 0
    cursor_detected_count: int = 0


# ---------------------------------------------------------------------------
# Cursor detection — motion based (three-frame AND technique)
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


def _find_cursor_two_frame(prev_gray, curr_gray, last_pos, y_min=0):
    """
    Fallback: find cursor from a two-frame diff when cursor is moving on a static bg.
    Uses the last_pos to distinguish old (near last_pos) from new position.
    Returns (x, y, area) or None.
    """
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20 or area > 600:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 50 or h > 50:
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
            continue  # skip browser chrome
        candidates.append((cx, cy, area))

    if not candidates or last_pos is None:
        return None

    # With two-frame diff, both old and new cursor positions appear.
    # The candidate NEAREST to last_pos is the OLD position (cursor just left).
    # We want the NEW position — the one that's farther from last_pos but still
    # within a reasonable range. Filter to candidates within 400px of last_pos.
    nearby = [(c, np.sqrt((c[0] - last_pos[0])**2 + (c[1] - last_pos[1])**2))
              for c in candidates]
    nearby = [(c, d) for c, d in nearby if d < 400]
    if not nearby:
        return None

    nearby.sort(key=lambda x: x[1])

    if len(nearby) >= 2:
        # Two candidates: nearest is old, second is new
        old_c, old_d = nearby[0]
        new_c, new_d = nearby[1]
        # Sanity: new shouldn't be absurdly far
        if new_d < 400:
            return new_c
    # Only one candidate — cursor barely moved, it's approximately here
    return nearby[0][0]


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


def track_cursor(video_path, sample_step=2, progress_callback=None):
    """
    Track cursor via three-frame AND-difference technique.
    The cursor appears in the intersection of diff(A,B) and diff(B,C) at position in B.
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

    # Read all sampled frames into a list of (frame_idx, gray)
    # For a 34-second video at 30fps sampling every 2nd frame, this is ~500 frames — fine for memory
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
    result.url_bar_region = url_region  # store for scene labeling
    print(f"  Loaded {len(frame_buffer)} sampled frames (browser chrome cutoff: y>{y_min})")

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    last_pos = None
    detected = 0
    scrolls = []

    # Process middle frames using three-frame windows
    for i in range(len(frame_buffer)):
        curr_idx, curr_gray = frame_buffer[i]
        timestamp = curr_idx / fps if fps > 0 else 0

        # --- Scroll detection (always check against previous frame) ---
        if i > 0:
            prev_idx, prev_gray = frame_buffer[i - 1]
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

        # --- Cursor detection ---
        cursor = None
        method = "hold"

        if i > 0:
            prev_idx, prev_gray = frame_buffer[i - 1]
            change = _frame_change_ratio(prev_gray, curr_gray)
        else:
            change = 1.0  # first frame, can't diff

        # Skip cursor detection on busy frames (scroll / page change)
        if change > 0.05:
            # Too much change — hold last position
            pass
        elif i >= 1 and i < len(frame_buffer) - 1:
            prev_idx, prev_gray = frame_buffer[i - 1]
            next_idx, next_gray = frame_buffer[i + 1]

            # Three-frame AND: cursor at current frame appears in both diffs
            d1 = cv2.absdiff(prev_gray, curr_gray)
            d2 = cv2.absdiff(curr_gray, next_gray)
            _, t1 = cv2.threshold(d1, 25, 255, cv2.THRESH_BINARY)
            _, t2 = cv2.threshold(d2, 25, 255, cv2.THRESH_BINARY)
            t1 = cv2.dilate(t1, dilate_kernel)
            t2 = cv2.dilate(t2, dilate_kernel)
            intersection = cv2.bitwise_and(t1, t2)
            intersection = cv2.morphologyEx(intersection, cv2.MORPH_OPEN, open_kernel)

            cursor = _find_cursor_in_intersection(intersection, last_pos, y_min=y_min)
            if cursor is not None:
                method = "three_frame"

            # Fallback: two-frame diff
            if cursor is None and last_pos is not None and change < 0.02:
                cursor = _find_cursor_two_frame(prev_gray, curr_gray, last_pos, y_min=y_min)
                if cursor is not None:
                    method = "two_frame"

        elif i >= 1:
            # Last frame — try two-frame only
            prev_idx, prev_gray = frame_buffer[i - 1]
            if change < 0.02 and last_pos is not None:
                cursor = _find_cursor_two_frame(prev_gray, curr_gray, last_pos, y_min=y_min)
                if cursor is not None:
                    method = "two_frame"

        if cursor is not None:
            cx, cy, _ = cursor
            # Confidence: three_frame is more reliable than two_frame
            conf = 0.8 if method == "three_frame" else 0.5
            last_pos = (cx, cy)
            detected += 1
            result.positions.append(CursorPosition(
                frame_idx=curr_idx, timestamp=round(timestamp, 3),
                x=cx, y=cy, confidence=conf, method=method,
            ))
        elif last_pos is not None:
            # Hold last known position with decaying confidence
            result.positions.append(CursorPosition(
                frame_idx=curr_idx, timestamp=round(timestamp, 3),
                x=last_pos[0], y=last_pos[1],
                confidence=0.1, method="hold",
            ))

        if progress_callback and i % 100 == 0:
            progress_callback(curr_idx, total_frames)

    result.scrolls = scrolls
    result.frames_analyzed = len(frame_buffer)
    result.cursor_detected_count = detected
    return result


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

    # Detect pauses
    pos_list = tracking_result.positions
    pause_frames = set()
    window = 3
    vel_list = [(p.frame_idx, velocities.get(p.frame_idx, 0)) for p in pos_list[1:]]
    for i in range(window, len(vel_list) - window):
        before = np.mean([vel_list[j][1] for j in range(i - window, i)])
        curr = vel_list[i][1]
        after = np.mean([vel_list[j][1] for j in range(i + 1, i + 1 + window)])
        if before > 50 and curr < 20 and after > 30:
            pause_frames.add(vel_list[i][0])

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
            if local < 0.7:
                prev_pos = pos_map.get(prev_fidx)
                if prev_pos:
                    dist = np.sqrt((pos.x - prev_pos.x) ** 2 + (pos.y - prev_pos.y) ** 2)
                    if dist < 60:
                        is_click = True
                        kind = "visual_change"
                        conf = max(0, 1.0 - local)

            # Signal 2: scene change (implies a preceding click)
            if glob < 0.6:
                sc = min(1.0, max(0, 1.0 - glob))
                if sc > conf:
                    is_click = True
                    kind = "scene_change"
                    conf = sc

            # Signal 3: cursor pause
            if fidx in pause_frames and not is_click:
                is_click = True
                kind = "pause_click"
                conf = 0.4

            if is_click and conf > 0.25:
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
        scroll_total = sum(s.pixels for s in tracking_result.scrolls if start <= s.frame_idx < end)

        scenes.append(Scene(
            start_frame=start, end_frame=end,
            start_time=round(start / fps, 2), end_time=round(end / fps, 2),
            screenshot_idx=len(scene_screenshots) - 1,
            click_count=click_count, scroll_total=scroll_total,
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
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{data}"


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
            notes += f'scroll {sc.scroll_total}px'
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
        scroll_note = f" | scroll {sc.scroll_total}px" if sc.scroll_total > 0 else ""
        sv = scene_visuals[i] if i < len(scene_visuals) else {}
        cb64 = _img_b64(sv.get("click_map", ""))
        tb64 = _img_b64(sv.get("trajectory", ""))
        ub64 = _img_b64(sv.get("url_bar", ""))
        url_html = f'<img src="{ub64}" class="url-bar">' if ub64 else ""
        per_scene_html += f'''
        <div class="scene-block">
            <h3>Scene {i+1}{bt}
                <span class="scene-meta">{sc.start_time:.1f}s &ndash; {sc.end_time:.1f}s
                &nbsp;|&nbsp; {dur:.1f}s &nbsp;|&nbsp; {sc.click_count} clicks{scroll_note}</span>
            </h3>
            {url_html}
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

    # Stage 1+2: Cursor tracking
    print("[1/6] Tracking cursor positions...")
    t0 = time.time()

    def progress(frame, total):
        print(f"       {frame/max(total,1)*100:.0f}% ({frame}/{total} frames)", end="\r")

    result = track_cursor(video_path, sample_step=sample_step, progress_callback=progress)
    n_scroll = len(result.scrolls)
    print(f"\n       Done. {result.cursor_detected_count}/{result.frames_analyzed} frames with cursor "
          f"({result.cursor_detected_count/max(result.frames_analyzed,1)*100:.0f}%), "
          f"{n_scroll} scroll events in {time.time()-t0:.1f}s")

    # Stage 3: Clicks
    print("[2/6] Detecting clicks...")
    t0 = time.time()
    detect_clicks(video_path, result)
    print(f"       Found {len(result.clicks)} clicks in {time.time()-t0:.1f}s")

    # Stage 4: Scenes
    print("[3/6] Detecting scenes/pages...")
    t0 = time.time()
    scene_screenshots, url_bar_crops = detect_scenes(video_path, result)
    bt = sum(1 for s in result.scenes if s.is_backtrack)
    print(f"       Found {len(result.scenes)} scenes ({bt} backtracks) in {time.time()-t0:.1f}s")

    # Stage 5: Visuals
    print("[4/6] Generating heatmap...")
    generate_heatmap(result, os.path.join(output_dir, "heatmap.png"))

    print("[5/6] Generating per-scene click maps & trajectories...")
    scene_visuals = generate_per_scene_visuals(result, scene_screenshots, url_bar_crops, output_dir)
    generate_journey_map(result, scene_screenshots, os.path.join(output_dir, "journey.png"))
    print(f"       Generated visuals for {len(scene_visuals)} scenes")

    # Stage 6: Report
    print("[6/6] Generating report...")
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
