"""
Transcription + speaker diarization pipeline for usability session recordings.
Uses faster-whisper (tiny) for transcription, MFCC clustering for speaker labels.

Output: transcript.txt and transcript.json saved next to each audio file.
Usage:
  python transcribe.py                  # all sessions
  python transcribe.py Saber            # single session by name
  python transcribe.py --speakers 2     # override speaker count (default: auto)
  python transcribe.py --redo-speakers  # re-run diarization on existing transcripts
"""

import json
import sys
import argparse
import numpy as np
import librosa
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from faster_whisper import WhisperModel


SESSIONS_DIR = Path(__file__).parent / "Session_Recordings"
TRANSCRIPTS_DIR = Path(__file__).parent / "transcripts"
MODEL_SIZE = "tiny"
DIARIZE_SR = 8000        # 8 kHz is sufficient for speaker MFCCs, keeps memory low
MIN_SPEAKERS = 2
MAX_SPEAKERS = 4         # hard cap — sessions never have more than 4 people


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h:
        return f"{h:02d}:{m:02d}:{s:05.2f}"
    return f"{m:02d}:{s:05.2f}"


def assign_speakers(segments: list, audio_path: Path, n_speakers: int | None) -> list:
    """Add 'speaker' field (P1, P2, ...) to each segment via MFCC clustering."""
    if len(segments) < 2:
        for seg in segments:
            seg["speaker"] = "P1"
        return segments

    print(f"  Loading audio for diarization at {DIARIZE_SR} Hz ...")
    audio, sr = librosa.load(str(audio_path), sr=DIARIZE_SR, mono=True)

    features = []
    for seg in segments:
        s = int(seg["start"] * sr)
        e = int(seg["end"] * sr)
        chunk = audio[s:e]
        if len(chunk) < sr * 0.1:
            features.append(np.zeros(40))
        else:
            mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=40)
            features.append(np.mean(mfcc, axis=1))

    X = StandardScaler().fit_transform(np.array(features))

    if n_speakers:
        # Manual override — use exactly what the user asked for
        k = min(n_speakers, MAX_SPEAKERS, len(segments))
        clustering = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = clustering.fit_predict(X)
        print(f"  Using user-specified k={k}")
    else:
        # Auto-detect best k in [MIN_SPEAKERS, MAX_SPEAKERS] via silhouette score
        max_k = min(MAX_SPEAKERS, len(segments))
        best_k, best_score, best_labels = MIN_SPEAKERS, -1, None
        for k in range(MIN_SPEAKERS, max_k + 1):
            clustering = AgglomerativeClustering(n_clusters=k, linkage="ward")
            trial_labels = clustering.fit_predict(X)
            score = silhouette_score(X, trial_labels)
            print(f"    k={k}: silhouette={score:.3f}")
            if score > best_score:
                best_k, best_score, best_labels = k, score, trial_labels
        labels = best_labels
        print(f"  Best k={best_k} (silhouette={best_score:.3f})")

    # Map cluster IDs → P1, P2, ... in order of first appearance
    speaker_map = {}
    counter = 1
    for label in labels:
        if label not in speaker_map:
            speaker_map[label] = f"P{counter}"
            counter += 1

    for seg, label in zip(segments, labels):
        seg["speaker"] = speaker_map[label]

    n_found = len(speaker_map)
    print(f"  Assigned {n_found} speaker(s): {', '.join(speaker_map.values())}")
    return segments


def session_number(audio_path: Path) -> str:
    """Extract session number from parent folder name, e.g. 'Session1_Saber' → '1'."""
    name = audio_path.parent.name
    return "".join(filter(str.isdigit, name.split("_")[0]))


def transcript_dir(audio_path: Path) -> Path:
    """Map Session_Recordings/Session1_Saber/ → transcripts/Session_1/"""
    num = session_number(audio_path)
    out = TRANSCRIPTS_DIR / f"Session_{num}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_outputs(audio_path: Path, segments: list) -> None:
    out_dir = transcript_dir(audio_path)
    num = session_number(audio_path)
    lines = [
        f"[{format_timestamp(s['start'])} --> {format_timestamp(s['end'])}]  "
        f"{s.get('speaker', '??')}:  {s['text']}"
        for s in segments
    ]
    (out_dir / f"transcript_{num}.txt").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / f"transcript_{num}.json").write_text(
        json.dumps({"audio": audio_path.name, "segments": segments}, indent=2),
        encoding="utf-8",
    )
    print(f"  Saved: transcript_{num}.txt, transcript_{num}.json")


def transcribe_file(
    audio_path: Path,
    model: WhisperModel,
    n_speakers: int | None,
    redo_speakers: bool,
) -> None:
    num = session_number(audio_path)
    json_out = transcript_dir(audio_path) / f"transcript_{num}.json"

    # Check what work is already done
    existing_segments = None
    has_speakers = False
    if json_out.exists():
        try:
            data = json.loads(json_out.read_text(encoding="utf-8"))
            existing_segments = data.get("segments", [])
            has_speakers = bool(existing_segments and "speaker" in existing_segments[0])
        except Exception:
            pass

    if has_speakers and not redo_speakers:
        print(f"  [skip] already transcribed + diarized")
        return

    # --- Transcription ---
    if existing_segments is None:
        print(f"  Transcribing: {audio_path.name}")
        segments_iter, info = model.transcribe(
            str(audio_path),
            language="en",
            beam_size=1,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        print(
            f"  Language: {info.language} (p={info.language_probability:.2f}), "
            f"duration: {info.duration:.1f}s"
        )
        segments = []
        for seg in segments_iter:
            entry = {"start": round(seg.start, 3), "end": round(seg.end, 3), "text": seg.text.strip()}
            segments.append(entry)
            print(f"    [{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}]  {seg.text.strip()}")
    else:
        print(f"  Reusing existing transcription ({len(existing_segments)} segments)")
        segments = existing_segments

    # --- Diarization ---
    segments = assign_speakers(segments, audio_path, n_speakers)

    write_outputs(audio_path, segments)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session", nargs="?", help="Filter by session name fragment")
    parser.add_argument("--speakers", type=int, default=None, metavar="N",
                        help="Number of speakers to detect (default: auto)")
    parser.add_argument("--redo-speakers", action="store_true",
                        help="Re-run diarization even if speakers already assigned")
    args = parser.parse_args()

    m4a_files = sorted(SESSIONS_DIR.glob("*/*.m4a"))
    if not m4a_files:
        print("No .m4a files found under Session_Recordings/")
        return

    if args.session:
        m4a_files = [f for f in m4a_files if args.session.lower() in f.parent.name.lower()]
        if not m4a_files:
            print(f"No sessions matching '{args.session}' found.")
            return

    print(f"Loading faster-whisper model '{MODEL_SIZE}' ...")
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    print(f"Model loaded. Processing {len(m4a_files)} file(s).\n")

    for audio_path in m4a_files:
        print(f"[{audio_path.parent.name}]")
        try:
            transcribe_file(audio_path, model, args.speakers, args.redo_speakers)
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
        print()

    print("Done.")


if __name__ == "__main__":
    main()
