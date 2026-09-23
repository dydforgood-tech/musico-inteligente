"""Exporta diagnóstico do ouvido em CSV para uma faixa real.

Exemplo: python tools/export_audio_diagnostics.py caminho.mp3 saida.csv --key "G# Major"
"""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.analysis.audio_analyzer import AudioAnalyzer
from app.audio.audio_loader import FileAudioSource
from app.music.theory import PITCH_CLASSES
from app.song.song import Song
from app.song.song_session import SongSession


def export_audio(audio_path: Path, output_path: Path, key: str = "--",
                 interval: float = 0.1, project: Path | None = None,
                 song_index: int = 0, seconds: float | None = None) -> dict:
    source = FileAudioSource(str(audio_path))
    sample_rate = source.get_sample_rate()
    analyzer = AudioAnalyzer(sample_rate=sample_rate, chunk_size=4096)
    session = None
    if project is not None:
        data = json.loads(project.read_text(encoding="utf-8"))
        songs = [song for setlist in data.get("setlists", [])
                 for song in setlist.get("songs", [])]
        if not 0 <= song_index < len(songs):
            raise IndexError(f"Índice da música fora do intervalo: {song_index}")
        session = SongSession(Song.from_dict(songs[song_index]))
        key = key if key != "--" else session.song.key
    if key != "--":
        analyzer.set_harmonic_expectation(key=key)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["seconds", "raw_note", "raw_pitch_hz", "pitch_confidence",
              "key", "key_confidence", "active_notes", "raw_chord",
              "raw_chord_confidence", "stable_chord", "stable_root",
              "stable_chord_confidence", "stable_chord_duration_s",
              "stable_chord_stale", "audio_activity", "bpm",
              "chart_section", "chart_chord", "position_confidence",
              "tracking_state", "performance_state", "harmonic_event",
              "harmonic_event_elapsed_beats"]
    fields += [f"chroma_{name}" for name in PITCH_CLASSES]
    count = 0
    chord_changes = 0
    last_chord = "--"
    try:
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            t = 0.0
            while t < min(source.get_duration(), seconds or source.get_duration()):
                chunk = source.get_chunk_at(int(t * sample_rate), 4096)
                ctx = analyzer.analyze_chunk(chunk, sample_rate, t, session=session)
                stable = ctx.smoothed_detected_chord
                if stable != last_chord and stable != "--":
                    chord_changes += 1
                last_chord = stable
                row = {
                    "seconds": round(t, 3), "raw_note": ctx.raw_note,
                    "raw_pitch_hz": round(ctx.raw_pitch_hz, 2),
                    "pitch_confidence": round(ctx.pitch_confidence, 3),
                    "key": key if key != "--" else ctx.key,
                    "key_confidence": 1.0 if key != "--" else round(ctx.key_confidence, 3),
                    "active_notes": " ".join(ctx.active_notes),
                    "raw_chord": ctx.raw_detected_chord,
                    "raw_chord_confidence": round(ctx.raw_chord_confidence, 3),
                    "stable_chord": stable, "stable_root": ctx.stable_chord_root,
                    "stable_chord_confidence": round(ctx.stable_chord_confidence, 3),
                    "stable_chord_duration_s": round(ctx.stable_chord_duration, 3),
                    "stable_chord_stale": int(ctx.stable_chord_stale),
                    "audio_activity": round(ctx.audio_activity, 5),
                    "bpm": round(ctx.bpm, 2),
                    "chart_section": ctx.current_section if session is not None else "--",
                    "chart_chord": (session.chart_position.current_chord
                                    if session is not None else "--"),
                    "position_confidence": round(ctx.position_confidence, 3),
                    "tracking_state": ctx.tracking_state,
                    "performance_state": ctx.performance_state,
                    "harmonic_event": ctx.harmonic_event_chord,
                    "harmonic_event_elapsed_beats": round(ctx.current_chord_elapsed_beats, 3),
                }
                row.update({f"chroma_{name}": round(float(value), 3)
                            for name, value in zip(PITCH_CLASSES, ctx.chroma_vector)})
                writer.writerow(row)
                count += 1
                t = count * interval
    finally:
        source.close()
    result = {"duration_s": round(t, 1), "frames": count,
              "stable_chord_entries": chord_changes, "output": str(output_path)}
    if session is not None:
        result["position_transitions"] = session.position_estimator.position_transition_log[:12]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--key", default="--")
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--song-index", type=int, default=0)
    parser.add_argument("--seconds", type=float)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval deve ser positivo")
    print(export_audio(args.audio, args.output, args.key, args.interval,
                       args.project, args.song_index, args.seconds))


if __name__ == "__main__":
    main()
