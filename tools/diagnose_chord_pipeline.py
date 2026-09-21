"""Gera um CSV por frame do pipeline harmônico para reproduzir acordes stale."""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.analysis.audio_analyzer import AudioAnalyzer
from app.analysis.chord_history import ChordStabilizer
from app.music.theory import PITCH_CLASSES


FIELDS = (
    "file", "timestamp", "pitch", "raw_chord", "raw_confidence",
    "stable_chord", "stable_root", "stable_quality", "stable_confidence",
    "candidate", "candidate_root",
    "candidate_confidence", "candidate_age_ms", "candidate_frames",
    "support_age_ms", "audio_activity", "expected_chart_chord", "chart_prior",
    "tracking_state", "position", "section",
)


def analyze_file(path: Path, chunk_size: int = 4096):
    audio, sample_rate = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1).astype(np.float32)
    analyzer = AudioAnalyzer(sample_rate=sample_rate, chunk_size=chunk_size)
    rows = []
    for start in range(0, max(0, len(audio) - chunk_size + 1), chunk_size):
        timestamp = start / float(sample_rate)
        context = analyzer.analyze_chunk(
            audio[start:start + chunk_size], sample_rate, timestamp)
        rows.append({
            "file": path.name,
            "timestamp": round(timestamp, 3),
            "pitch": context.note,
            "raw_chord": context.raw_detected_chord,
            "raw_confidence": round(context.raw_chord_confidence, 3),
            "stable_chord": context.smoothed_detected_chord,
            "stable_root": context.stable_chord_root,
            "stable_quality": context.stable_chord_quality,
            "stable_confidence": round(context.stable_chord_confidence, 3),
            "candidate": context.chord_candidate,
            "candidate_root": context.chord_candidate_root,
            "candidate_confidence": round(context.chord_candidate_confidence, 3),
            "candidate_age_ms": round(context.chord_candidate_age_ms, 1),
            "candidate_frames": context.chord_candidate_frames,
            "support_age_ms": round(context.current_chord_support_age_ms, 1),
            "audio_activity": round(context.audio_activity, 5),
            "expected_chart_chord": context.expected_chart_chord,
            "chart_prior": "ON" if context.chart_prior_enabled else "OFF",
            "tracking_state": context.tracking_state,
            "position": f"{context.bar}:{context.beat}",
            "section": context.current_section,
        })
    return rows


def longest_unsupported_f_sharp(rows):
    longest = current = 0
    for row in rows:
        unsupported = (row["stable_chord"] == "F#" and
                       row["raw_chord"] not in ("F#", "--"))
        current = current + 1 if unsupported else 0
        longest = max(longest, current)
    return longest


def regression_chroma(root: str):
    chroma = np.zeros(12, dtype=np.float32)
    root_index = PITCH_CLASSES.index(root)
    for strength, interval in zip((1.0, .9, .8), (0, 4, 7)):
        chroma[(root_index + interval) % 12] = strength
    return chroma


def write_regression_comparison(path: Path):
    """Compara a regra consecutiva anterior com o estabilizador atual."""
    variants = (
        ("A", "A", "A"), ("A7", "A", "A"), ("A", "A", "A"),
        ("E/A", "E", "A"), ("A", "A", "A"),
        ("Asus4", "A", "A"), ("A", "A", "A"),
    )
    sequence = [(t, "F#", "F#", "F#", .90)
                for t in (0.0, .05, .10, .15)]
    sequence += [(.20 + index * .10, *variants[index % len(variants)], .62)
                 for index in range(50)]
    fixed = ChordStabilizer(min_confidence=.4, confirmation_time=.15)
    legacy_stable = "--"
    legacy_candidate = None
    legacy_count = 0
    rows = []
    for timestamp, symbol, root, bass, confidence in sequence:
        if legacy_stable == "--":
            legacy_stable = symbol
        elif symbol == legacy_stable:
            legacy_candidate = None
            legacy_count = 0
        elif legacy_candidate != symbol:
            legacy_candidate = symbol
            legacy_count = 1
        else:
            legacy_count += 1
            if legacy_count >= 3:
                legacy_stable = legacy_candidate
                legacy_candidate = None
                legacy_count = 0
        fixed_stable = fixed.process(
            symbol, confidence, timestamp, root=root, quality="major",
            bass_note=bass, chroma_vector=regression_chroma("A" if root != "F#" else "F#"),
            audio_activity=.08)
        rows.append({
            "timestamp": round(timestamp, 3), "raw_chord": symbol,
            "raw_confidence": confidence, "legacy_stable": legacy_stable,
            "fixed_stable": fixed_stable,
            "fixed_confidence": round(fixed.chord_confidence, 3),
            "candidate": fixed.candidate_chord,
            "candidate_root": fixed.candidate_root,
            "candidate_confidence": round(fixed.candidate_confidence, 3),
            "candidate_age_ms": round(fixed.candidate_duration * 1000.0, 1),
            "candidate_frames": fixed.candidate_frame_count,
            "support_age_ms": round(fixed.time_since_current_chord_support * 1000.0, 1),
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    first_fixed_exit = next((row["timestamp"] for row in rows
                             if row["fixed_stable"] != "F#"), None)
    print({"regression_csv": str(path),
           "legacy_final": rows[-1]["legacy_stable"],
           "fixed_final": rows[-1]["fixed_stable"],
           "fixed_left_f_sharp_at": first_fixed_exit})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path("docs/diagnostics/chord_stabilizer_audio_log.csv"))
    parser.add_argument("--regression-output", type=Path,
                        default=Path("docs/diagnostics/chord_stabilizer_regression_log.csv"))
    args = parser.parse_args()

    all_rows = []
    for path in args.files:
        rows = analyze_file(path)
        all_rows.extend(rows)
        print({
            "file": path.name,
            "frames": len(rows),
            "raw": Counter(row["raw_chord"] for row in rows).most_common(6),
            "stable": Counter(row["stable_chord"] for row in rows).most_common(6),
            "longest_unsupported_f_sharp_frames": longest_unsupported_f_sharp(rows),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"CSV={args.output}")
    write_regression_comparison(args.regression_output)


if __name__ == "__main__":
    main()
