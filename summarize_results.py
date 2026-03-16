"""
Summarize all tracker raw_data.json outputs into a single CSV.
One row per video (session + task).
"""

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent / "Session_Recordings"
OUT_CSV = Path(__file__).parent / "results_summary.csv"

FIELDS = [
    "participant",
    "session",
    "task",
    "video_file",
    "duration_s",
    "fps",
    "resolution",
    "frames_analyzed",
    "cursor_detected_count",
    "detection_rate_pct",
    "total_clicks",
    "total_scrolls",
    "scroll_up_count",
    "scroll_down_count",
    "total_scroll_pixels",
    "scroll_time_s",
    "total_scenes",
    "backtrack_scenes",
]

def _scroll_time_s(scrolls, sample_step, fps, gap_s=0.5):
    """Mirror of tracker._scroll_time_s — works on raw dicts from JSON."""
    if not scrolls:
        return 0.0
    sample_interval = sample_step / fps if fps > 0 else 0.0
    timestamps = sorted(s.get("timestamp", 0) for s in scrolls)
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


rows = []

for json_path in sorted(ROOT.rglob("raw_data.json")):
    with open(json_path) as f:
        d = json.load(f)

    # Parse participant + session + task from the folder structure
    # Expected: Session_Recordings/SessionN_Name/TaskN/output_*/raw_data.json
    parts = json_path.parts
    session_folder = next((p for p in parts if re.match(r"Session\d+_", p)), "")
    task_folder = next((p for p in parts if re.match(r"Task\d+", p) and "output" not in p), "")

    m = re.match(r"Session(\d+)_(.+)", session_folder)
    session_num = m.group(1) if m else ""
    participant = m.group(2) if m else session_folder

    tm = re.match(r"Task(\d+)", task_folder)
    task_num = tm.group(1) if tm else task_folder

    v = d["video"]
    a = d["analysis"]
    clicks = d.get("clicks", [])
    scrolls = d.get("scrolls", [])
    scenes = d.get("scenes", [])

    scroll_up = sum(1 for s in scrolls if s.get("direction") == "up")
    scroll_down = sum(1 for s in scrolls if s.get("direction") == "down")
    scroll_pixels = sum(abs(s.get("pixels", 0)) for s in scrolls)
    scroll_time = _scroll_time_s(scrolls, a.get("sample_step", 2), v.get("fps", 30))
    backtracks = sum(1 for sc in scenes if sc.get("is_backtrack"))

    rows.append({
        "participant": participant,
        "session": session_num,
        "task": task_num,
        "video_file": Path(v["path"]).name,
        "duration_s": round(v["duration"], 2),
        "fps": v["fps"],
        "resolution": f"{v['width']}x{v['height']}",
        "frames_analyzed": a["frames_analyzed"],
        "cursor_detected_count": a["cursor_detected_count"],
        "detection_rate_pct": round(a["detection_rate"] * 100, 1),
        "total_clicks": len(clicks),
        "total_scrolls": len(scrolls),
        "scroll_up_count": scroll_up,
        "scroll_down_count": scroll_down,
        "total_scroll_pixels": scroll_pixels,
        "scroll_time_s": scroll_time,
        "total_scenes": len(scenes),
        "backtrack_scenes": backtracks,
    })

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {OUT_CSV}")
for r in rows:
    print(f"  {r['participant']:12s} Task{r['task']}  {r['duration_s']:7.1f}s  "
          f"{r['detection_rate_pct']:5.1f}% detected  "
          f"{r['total_clicks']} clicks  {r['total_scrolls']} scrolls  {r['total_scenes']} scenes")
