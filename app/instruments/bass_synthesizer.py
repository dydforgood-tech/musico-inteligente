"""Sintetizador Local de Baixo Elétrico (BassSynthesizer).

Responsável por sintetizar som de contrabaixo elétrico puro e quente em tempo real:
- Síntese acústica aditiva com harmônicos calibrados (fundamental, 2º e 3º harmônicos).
- Pluck transient (ataque percussivo característico da corda dedilhada ou palhetada).
- Envelope ADSR suave com decaimento natural de instrumento de corda.
- Gerenciamento de vozes polifônicas/monofônicas thread-safe.
- Renderização em blocos para mixagem contínua no AudioPlayer com zero latência.
"""

import threading
from typing import List, Optional
import numpy as np

from app.music.theory import midi_to_hz
from app.music.constants import BASS_DEFAULT_VOLUME


class ActiveVoice:
    """Representa uma nota em execução no sintetizador."""

    def __init__(self, freq: float, velocity: int, duration_sec: float, sample_rate: int):
        self.freq = freq
        self.velocity = float(velocity) / 127.0
        self.duration_samples = int(duration_sec * sample_rate)
        self.release_samples = int(0.035 * sample_rate)  # 35 ms de release suave
        self.total_samples = self.duration_samples + self.release_samples
        self.current_sample = 0
        self.is_finished = False

        # Pré-cálculo do envelope temporal ADSR
        t = np.arange(self.total_samples, dtype=np.float32) / float(sample_rate)

        # 1. Ataque rápido (10 ms)
        att_len = max(1, int(0.010 * sample_rate))
        attack = np.ones(self.total_samples, dtype=np.float32)
        if att_len < self.total_samples:
            attack[:att_len] = np.linspace(0.0, 1.0, att_len, dtype=np.float32)

        # 2. Decaimento natural de corda com sustain
        decay = np.exp(-t / 0.40) * 0.70 + 0.30

        # 3. Release suave ao final da duração
        release = np.ones(self.total_samples, dtype=np.float32)
        if self.duration_samples < self.total_samples:
            rel_len = self.total_samples - self.duration_samples
            release[self.duration_samples:] = np.linspace(1.0, 0.0, rel_len, dtype=np.float32)

        envelope = attack * decay * release

        # Harmônicos calibrados de contrabaixo elétrico:
        # Fundamental (1.0), 2º harmônico quente (0.45), 3º harmônico com punch (0.18), 4º harmônico (0.06)
        w1 = 2.0 * np.pi * self.freq
        wave = (
            1.00 * np.sin(w1 * t) +
            0.45 * np.sin(2.0 * w1 * t) +
            0.18 * np.sin(3.0 * w1 * t) +
            0.06 * np.sin(4.0 * w1 * t)
        )

        # Transiente de palheta/dedilhado nos primeiros 12 ms
        pluck_len = min(self.total_samples, int(0.012 * sample_rate))
        if pluck_len > 0:
            pluck_t = t[:pluck_len]
            pluck = np.sin(5.0 * w1 * pluck_t) * np.exp(-pluck_t / 0.003) * 0.30
            wave[:pluck_len] += pluck

        # Sinal completo da nota normalizado e amplificado pela dinâmica (velocity)
        raw_signal = wave * envelope * (self.velocity * 0.85)
        self.signal = raw_signal.astype(np.float32)

    def render(self, frames: int) -> np.ndarray:
        """Extrai a fatia do sinal para o bloco atual."""
        if self.is_finished or self.current_sample >= self.total_samples:
            self.is_finished = True
            return np.zeros(frames, dtype=np.float32)

        rem = self.total_samples - self.current_sample
        take = min(frames, rem)

        out = np.zeros(frames, dtype=np.float32)
        out[:take] = self.signal[self.current_sample:self.current_sample + take]

        self.current_sample += take
        if self.current_sample >= self.total_samples:
            self.is_finished = True

        return out


class BassSynthesizer:
    """Sintetizador de Baixo Elétrico com mixagem em tempo real e controle de dinâmica."""

    def __init__(self, sample_rate: int = 44100, volume: float = BASS_DEFAULT_VOLUME):
        self._sample_rate = sample_rate
        self._volume = volume
        self._enabled = True
        self._lock = threading.Lock()
        self._voices: List[ActiveVoice] = []

    @property
    def volume(self) -> float:
        return self._volume

    @volume.setter
    def volume(self, val: float) -> None:
        self._volume = max(0.0, min(1.0, float(val)))

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, val: bool) -> None:
        self._enabled = bool(val)

    def set_sample_rate(self, sr: int) -> None:
        with self._lock:
            self._sample_rate = sr
            self._voices.clear()

    def clear(self) -> None:
        """Interrompe todas as vozes ativas."""
        with self._lock:
            self._voices.clear()

    def trigger_note(self, midi_note: int, velocity: int = 100, duration: float = 0.5) -> None:
        """Dispara uma nova nota de baixo imediatamente."""
        if not self._enabled or midi_note <= 0:
            return

        freq = midi_to_hz(midi_note)
        if freq < 20.0 or freq > 1000.0:
            return

        voice = ActiveVoice(
            freq=freq,
            velocity=velocity,
            duration_sec=duration,
            sample_rate=self._sample_rate
        )

        with self._lock:
            # Monofonia com corte suave da nota anterior (estilo contrabaixo)
            # Mantém no máximo 2 vozes simultâneas durante transições rápidas
            if len(self._voices) >= 2:
                self._voices = self._voices[-1:]
            self._voices.append(voice)

    def render_chunk(self, frames: int, sample_rate: int, current_pos: float = 0.0) -> np.ndarray:
        """Gera um bloco de áudio estéreo (frames, 2) pronto para mixagem no AudioPlayer."""
        if not self._enabled or self._volume <= 0.0:
            return np.zeros((frames, 2), dtype=np.float32)

        if sample_rate != self._sample_rate:
            self.set_sample_rate(sample_rate)

        mono_mix = np.zeros(frames, dtype=np.float32)

        with self._lock:
            active_voices = []
            for voice in self._voices:
                chunk = voice.render(frames)
                mono_mix += chunk
                if not voice.is_finished:
                    active_voices.append(voice)
            self._voices = active_voices

        # Aplica volume master do baixo com proteção estrita contra saturação/clipping
        mono_mix = np.clip(mono_mix * self._volume, -1.0, 1.0)

        # Retorna estéreo (L e R idênticos para baixo centrado)
        return np.column_stack([mono_mix, mono_mix]).astype(np.float32)

    def synthesize_note_to_array(
        self,
        midi_note: int,
        duration: float,
        velocity: int = 100,
        sr: int = 44100
    ) -> np.ndarray:
        """Método utilitário para renderizar uma nota completa em array NumPy (ideal para testes)."""
        freq = midi_to_hz(midi_note)
        voice = ActiveVoice(freq, velocity, duration, sr)
        return voice.signal
