"""Extrator de Cromagrama (12 Classes de Notas Musicais).

Calcula a distribuição espectral de energia sobre as 12 notas da escala temperada
[C, C#, D, D#, E, F, F#, G, G#, A, A#, B], essencial para visualização harmônica
e detecção de acordes.
"""

from typing import List, Tuple
import numpy as np

from app.music.theory import PITCH_CLASSES


class ChromaExtractor:
    """Extrai o perfil das 12 classes de pitch (Cromagrama).

    Dois métodos disponíveis:
      - ``"fft"`` (padrão): acumula a energia espectral por classe de pitch. Rápido e
        estável, porém sensível a vazamento de harmônicos (um parcial em 3f é contado
        na classe de 3f, não na do fundamental f).
      - ``"harmonic"``: **soma harmônica** (harmonic salience). Para cada fundamental
        candidato soma a energia dos seus harmônicos, atribuindo os parciais ao fundamental
        correto. Suprime o "vazamento" de overtones (ex.: o Si-fantasma do 3º harmônico do
        Mi num acorde de Dó maior), o que torna viável distinguir tétrades (maj7/m7) de
        tríades puras. Custo levemente maior (~1-2 ms por bloco).
    """

    def __init__(self, fmin: float = 30.0, fmax: float = 4200.0, method: str = "fft",
                 n_harmonics: int = 5, harmonic_decay: float = 0.8):
        self._fmin = fmin
        self._fmax = fmax
        self._method = method if method in ("fft", "harmonic") else "fft"
        self._n_harmonics = max(1, int(n_harmonics))
        self._harmonic_decay = float(harmonic_decay)

    @property
    def method(self) -> str:
        return self._method

    def extract(self, audio_chunk: np.ndarray, sample_rate: int) -> np.ndarray:
        """Retorna um array NumPy de 12 floats normalizado [0.0 a 1.0] (C, C#, ... B)."""
        if len(audio_chunk) < 256 or sample_rate <= 0:
            return np.zeros(12, dtype=np.float32)

        # Verificar silêncio
        rms = float(np.sqrt(np.mean(audio_chunk ** 2)))
        if rms < 0.005:
            return np.zeros(12, dtype=np.float32)

        # Janelamento e FFT (comum aos dois métodos)
        x = audio_chunk - np.mean(audio_chunk)
        xw = x * np.hanning(len(x))
        # O método harmônico usa maior resolução espectral (8192); o FFT preserva o piso
        # histórico (4096) para não alterar a calibração existente do pipeline padrão.
        floor = 8192 if self._method == "harmonic" else 4096
        n_fft = max(floor, 2 ** int(np.ceil(np.log2(len(xw)))))
        mag = np.abs(np.fft.rfft(xw, n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)

        if self._method == "harmonic":
            return self._extract_harmonic(mag, freqs)
        return self._extract_fft(mag, freqs)

    def _extract_fft(self, mag: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        """Cromagrama por acumulação direta de energia espectral (método clássico)."""
        chroma = np.zeros(12, dtype=np.float32)
        valid = (freqs >= self._fmin) & (freqs <= self._fmax)
        valid_freqs = freqs[valid]
        valid_mags = mag[valid]
        if len(valid_freqs) == 0:
            return chroma

        # MIDI = 69 + 12 * log2(f / 440)
        midi_notes = 69.0 + 12.0 * np.log2(valid_freqs / 440.0)
        pitch_classes = (np.round(midi_notes).astype(int)) % 12
        energies = valid_mags ** 2
        np.add.at(chroma, pitch_classes, energies)

        max_energy = np.max(chroma)
        if max_energy > 0:
            chroma /= max_energy
        return chroma.astype(np.float32)

    def _extract_harmonic(self, mag: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        """Cromagrama por soma harmônica (atribui parciais ao fundamental correto)."""
        chroma = np.zeros(12, dtype=np.float32)
        # Grade de fundamentais candidatos (~1 Hz), limitada à faixa musical de graves/médios
        f_lo = max(self._fmin, 50.0)
        f_hi = min(self._fmax, 2100.0)
        if f_hi <= f_lo:
            return chroma
        cand = np.arange(f_lo, f_hi, 1.0, dtype=np.float64)

        # Salience de cada fundamental = soma dos seus harmônicos com decaimento
        salience = np.zeros(len(cand), dtype=np.float64)
        for h in range(1, self._n_harmonics + 1):
            weight = self._harmonic_decay ** (h - 1)
            salience += weight * np.interp(cand * h, freqs, mag, left=0.0, right=0.0)

        midi_notes = 69.0 + 12.0 * np.log2(cand / 440.0)
        pitch_classes = (np.round(midi_notes).astype(int)) % 12
        np.add.at(chroma, pitch_classes, salience ** 2)

        max_energy = np.max(chroma)
        if max_energy > 0:
            chroma /= max_energy
        return chroma.astype(np.float32)

    def get_top_notes(self, chroma: np.ndarray, top_n: int = 3) -> List[Tuple[str, float]]:
        """Retorna as top_n notas mais ativas no cromagrama."""
        indices = np.argsort(-chroma)[:top_n]
        return [(PITCH_CLASSES[i], float(chroma[i])) for i in indices if chroma[i] > 0.15]
