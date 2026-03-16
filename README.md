# Video Mouse Tracker

Retroactively tracks mouse cursor movements and clicks from MP4 screen recordings. Produces heatmaps, click maps, trajectory visualizations, journey maps, and a standalone HTML report.

Built for analyzing usability testing sessions — supports multiple cursor types, handles varying recording conditions, and produces per-scene breakdowns with backtrack detection.

## How It Works

The tracker uses a multi-method detection pipeline:

1. **Template matching** — matches cursor templates (arrow, I-beam, pointer) with alpha-mask support
2. **Color detection** — finds colored cursors (e.g. red accessibility cursors) in HSV space
3. **Motion detection** — three-frame AND-difference isolates cursor movement from page content
4. **Kalman filtering** — smooths trajectory and predicts position during detection gaps
5. **Scene detection** — identifies page changes via structural similarity, with backtrack identification

Detection methods are combined and validated against each other to reduce false positives.

## Features

- **Multi-method cursor tracking** — template matching, color detection, and motion analysis
- **Click detection** using local visual changes, scene transitions, and cursor pause analysis
- **Scroll detection** via strip cross-correlation (direction + pixel displacement)
- **Scene/page detection** with backtrack identification
- **Per-scene visualizations** — click maps, trajectory overlays, animated GIFs, URL bar crops
- **Heatmap** of cursor positions across all pages
- **Journey map** timeline showing scene durations and interactions
- **Standalone HTML report** with all images embedded as base64

## Requirements

- Python 3.10+
- OpenCV (`opencv-python`)
- NumPy
- Matplotlib

```
pip install opencv-python numpy matplotlib
```

Additional requirements for transcription (`transcribe.py`):
```
pip install faster-whisper librosa scikit-learn
```

## Usage

### Single video

```
python tracker.py <video_file.mp4>
```

### Batch processing

Process all session recordings at once (skips already-processed videos):

```
python batch_process.py
```

Expects videos in `Session_Recordings/SessionN_Name/TaskN/TaskN_Name.mp4`.

### Regenerate reports

Re-generate HTML reports from existing `raw_data.json` without re-tracking:

```
python regen_reports.py
```

### Summarize results

Aggregate all tracking results into a single CSV:

```
python summarize_results.py
```

### Transcribe session audio

Transcribe `.m4a` audio files with speaker diarization:

```
python transcribe.py              # all sessions
python transcribe.py Saber        # single session by name
python transcribe.py --speakers 2 # override speaker count
```

## Output

Output is saved to an `output_<video_name>/` directory alongside the video:

| File | Description |
|------|-------------|
| `report.html` | Standalone HTML report with all visualizations |
| `heatmap.png` | Screen position heatmap (all pages combined) |
| `journey.png` | Timeline visualization of scenes |
| `raw_data.json` | Raw tracking data (positions, clicks, scrolls, scenes) |
| `scenes/` | Per-scene click maps, trajectories, GIFs, and URL bar crops |

## Project Structure

| File | Description |
|------|-------------|
| `tracker.py` | Core tracking engine and report generator |
| `batch_process.py` | Batch runner for processing all session videos |
| `regen_reports.py` | Regenerate HTML reports from existing raw data |
| `summarize_results.py` | Aggregate results into `results_summary.csv` |
| `transcribe.py` | Audio transcription with speaker diarization |
| `debug_frames/` | Sample frames and cursor crops for development |
