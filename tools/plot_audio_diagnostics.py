"""Desenha pitch bruto, notas ativas e raiz estabilizada a partir do CSV."""
import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.music.theory import PITCH_CLASSES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file", type=Path)
    parser.add_argument("image_file", type=Path)
    parser.add_argument("--seconds", type=float, default=45.0)
    args = parser.parse_args()
    with args.csv_file.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle)
                if float(row["seconds"]) <= args.seconds]
    pitch_points, active_points, roots = [], [], []
    for row in rows:
        second = float(row["seconds"])
        raw = row["raw_note"].rstrip("0123456789")
        if raw in PITCH_CLASSES:
            pitch_points.append((second, PITCH_CLASSES.index(raw)))
        for note in row["active_notes"].split():
            if note in PITCH_CLASSES:
                active_points.append((second, PITCH_CLASSES.index(note)))
        root = row["stable_root"]
        if root in PITCH_CLASSES:
            roots.append((second, PITCH_CLASSES.index(root)))

    fig, axes = plt.subplots(3, 1, figsize=(15, 8), sharex=True)
    for axis, points, label, color in zip(
            axes, (pitch_points, active_points, roots),
            ("Pitch bruto (uma nota)", "Notas ativas simultâneas", "Raiz do acorde estabilizado"),
            ("#93a3b8", "#00a6a6", "#e87924")):
        if points:
            axis.scatter(*zip(*points), s=8 if axis is not axes[2] else 12,
                         alpha=.65, c=color)
        axis.set_yticks(range(12), PITCH_CLASSES)
        axis.set_ylim(-.6, 11.6)
        axis.set_ylabel(label)
        axis.grid(alpha=.2)
    axes[-1].set_xlabel("Tempo do áudio (s)")
    fig.suptitle(f"Diagnóstico harmônico — primeiros {args.seconds:g} segundos")
    fig.tight_layout()
    args.image_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.image_file, dpi=150)
    print(args.image_file)


if __name__ == "__main__":
    main()
