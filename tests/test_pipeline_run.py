"""Script de validação do pipeline completo com o áudio padrão."""

import os
import sys

# Garante inclusão da raiz do projeto no path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from app.audio.audio_loader import FileAudioSource
from app.analysis.audio_analyzer import AudioAnalyzer

def run():
    wav_path = os.path.abspath("test_song_120bpm.wav")
    if not os.path.exists(wav_path):
        print("Gerando test_song_120bpm.wav...")
        from app.utils.audio_generator import generate_test_song
        generate_test_song(wav_path)

    source = FileAudioSource(wav_path)
    sr = source.get_sample_rate()
    analyzer = AudioAnalyzer(sample_rate=sr, chunk_size=4096)

    # Pré-analisar BPM com os primeiros 10 segundos
    sub_y = source.mono_data[:sr * 10]
    bpm = analyzer.pre_analyze_track(sub_y, sr)

    print(f"BPM Detectado: {bpm:.1f} BPM", flush=True)
    print("=" * 75, flush=True)

    # Testar pontos correspondentes a C, G, Am, F
    test_points = [
        (1.0, "C"),
        (3.0, "G"),
        (5.0, "Am"),
        (7.0, "F")
    ]

    for t, expected in test_points:
        frame = int(t * sr)
        chunk = source.get_chunk_at(frame, 4096)
        ctx = analyzer.analyze_chunk(chunk, sr, t)
        print(f"Tempo: {t:4.1f}s | Esperado: {expected:2s} | Acorde: {ctx.current_chord:4s} ({ctx.inversion:4s}) | Tom: {ctx.current_key:10s} ({int(ctx.key_confidence*100)}%) | Beat: {ctx.beat}/4", flush=True)

    print("=" * 75, flush=True)
    print("Historico de Acordes Recente:", flush=True)
    print(analyzer.chord_history.get_summary_text(), flush=True)
    source.close()

if __name__ == "__main__":
    run()
