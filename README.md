# Video Mouse Tracker

Retroactively tracks mouse cursor movements and clicks from MP4 screen recordings. Produces heatmaps, click maps, trajectory visualizations, journey maps, and a standalone HTML report.

## How It Works

The tracker uses a **three-frame AND-difference technique** to detect cursor movement in screen recordings:

1. For three consecutive frames A, B, C, the cursor at frame B appears in the intersection of `diff(A, B)` and `diff(B, C)`
2. Falls back to two-frame differencing when three-frame detection fails
3. Filters out browser chrome, animated page elements, and scroll artifacts

## Features

- **Cursor tracking** via motion-based detection (no template matching needed)
- **Click detection** using local visual changes, scene transitions, and cursor pause analysis
- **Scroll detection** via strip cross-correlation
- **Scene/page detection** with backtrack identification
- **Per-scene visualizations** — click maps and trajectory overlays on screenshots
- **Heatmap** of cursor positions across all pages
- **Journey map** timeline showing scene durations and interactions
- **Standalone HTML report** with all images embedded

## Requirements

- Python 3.10+
- OpenCV (`opencv-python`)
- NumPy
- Matplotlib

```
pip install opencv-python numpy matplotlib
```

## Usage

```
python tracker.py <video_file.mp4>
```

Output is saved to an `output_<video_name>/` directory alongside the video, containing:

| File | Description |
|------|-------------|
| `report.html` | Standalone HTML report with all visualizations |
| `heatmap.png` | Screen position heatmap (all pages combined) |
| `journey.png` | Timeline visualization of scenes |
| `raw_data.json` | Raw tracking data (positions, clicks, scrolls, scenes) |
| `scenes/` | Per-scene click maps, trajectories, and URL bar crops |
