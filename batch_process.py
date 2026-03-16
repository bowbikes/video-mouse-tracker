"""
Batch processor for video mouse tracker.
Finds all MP4 files inside Task* folders under Session_Recordings/
and runs tracker.py on each one, skipping any that already have
an output_* directory in the same Task folder.
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
SESSIONS_DIR = ROOT / "Session_Recordings"
TRACKER = ROOT / "tracker.py"


def find_task_videos():
    """Find all MP4 files inside Task* subdirectories."""
    return sorted(SESSIONS_DIR.glob("*/Task*/*.mp4"))


def is_already_processed(video_path: Path) -> bool:
    """Check if an output directory already exists for this video in the Task folder."""
    stem = video_path.stem[:50]
    output_dir = video_path.parent / f"output_{stem}"
    return output_dir.exists()


def main():
    if not SESSIONS_DIR.exists():
        print(f"Session_Recordings directory not found: {SESSIONS_DIR}")
        sys.exit(1)

    videos = find_task_videos()

    if not videos:
        print("No MP4 files found in Task folders. Nothing to process.")
        sys.exit(0)

    already_done = [v for v in videos if is_already_processed(v)]
    to_process = [v for v in videos if not is_already_processed(v)]

    print(f"Found {len(videos)} task video(s) total")
    if already_done:
        print(f"  Skipping {len(already_done)} already processed:")
        for v in already_done:
            print(f"    - {v.relative_to(SESSIONS_DIR)}")
    print(f"  Processing {len(to_process)} video(s):\n")

    if not to_process:
        print("All videos already processed. Done!")
        return

    for i, video in enumerate(to_process, 1):
        rel = video.relative_to(SESSIONS_DIR)
        print(f"[{i}/{len(to_process)}] Processing: {rel}")
        start = time.time()

        result = subprocess.run(
            [sys.executable, str(TRACKER), str(video)],
            cwd=str(ROOT),
        )

        elapsed = time.time() - start
        if result.returncode == 0:
            print(f"  Done in {elapsed:.0f}s\n")
        else:
            print(f"  FAILED (exit code {result.returncode}) after {elapsed:.0f}s\n")

    print("Batch processing complete.")


if __name__ == "__main__":
    main()
