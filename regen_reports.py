"""
Regenerate report.html for all already-processed sessions using existing
raw_data.json and scene images — no re-tracking needed.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from tracker import (
    TrackingResult, CursorPosition, ClickEvent, ScrollEvent, Scene,
    generate_report, _scroll_time_s,
)


def load_tracking_result(json_path: Path) -> TrackingResult:
    with open(json_path) as f:
        d = json.load(f)

    v = d["video"]
    a = d["analysis"]

    positions = [CursorPosition(**p) for p in d.get("positions", [])]
    clicks    = [ClickEvent(**c)     for c in d.get("clicks", [])]
    scrolls   = [ScrollEvent(**s)    for s in d.get("scrolls", [])]

    sample_step = a.get("sample_step", 2)
    fps         = v["fps"]

    scenes = []
    for sc in d.get("scenes", []):
        scene_scrolls = [s for s in scrolls
                         if sc["start_frame"] <= s.frame_idx < sc["end_frame"]]
        scroll_time = _scroll_time_s(scene_scrolls, sample_step, fps)
        scenes.append(Scene(
            start_frame     = sc["start_frame"],
            end_frame       = sc["end_frame"],
            start_time      = sc["start_time"],
            end_time        = sc["end_time"],
            screenshot_idx  = sc.get("screenshot_idx", 0),
            click_count     = sc.get("click_count", 0),
            scroll_total    = sc.get("scroll_total", 0),
            scroll_time_s   = scroll_time,
            is_backtrack    = sc.get("is_backtrack", False),
            similar_scene_idx = sc.get("similar_scene_idx", -1),
        ))

    return TrackingResult(
        video_path           = v["path"],
        width                = v["width"],
        height               = v["height"],
        fps                  = fps,
        total_frames         = v["total_frames"],
        duration             = v["duration"],
        sample_step          = sample_step,
        positions            = positions,
        clicks               = clicks,
        scrolls              = scrolls,
        scenes               = scenes,
        frames_analyzed      = a["frames_analyzed"],
        cursor_detected_count= a["cursor_detected_count"],
    )


def build_scene_visuals(output_dir: Path, n_scenes: int) -> list:
    scenes_dir = output_dir / "scenes"
    visuals = []
    for i in range(1, n_scenes + 1):
        tag = f"{i:02d}"
        vis = {
            "click_map":  str(scenes_dir / f"scene_{tag}_clicks.png"),
            "trajectory": str(scenes_dir / f"scene_{tag}_trajectory.png"),
            "gif":        str(scenes_dir / f"scene_{tag}.gif"),
            "url_bar":    str(scenes_dir / f"scene_{tag}_url.png"),
        }
        visuals.append(vis)
    return visuals


def main():
    json_paths = sorted(
        ROOT.glob("Session_Recordings/*/Task*/output_*/raw_data.json")
    )

    if not json_paths:
        print("No raw_data.json files found.")
        return

    print(f"Regenerating {len(json_paths)} report(s)...\n")

    for jp in json_paths:
        output_dir = jp.parent
        rel = jp.relative_to(ROOT)
        try:
            result = load_tracking_result(jp)
            scene_visuals = build_scene_visuals(output_dir, len(result.scenes))
            report_path = generate_report(result, str(output_dir), [], scene_visuals)
            print(f"  OK  {rel.parent}  ({len(result.scenes)} scenes, "
                  f"{len(result.scrolls)} scrolls)")
        except Exception as e:
            print(f"  FAIL {rel}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
