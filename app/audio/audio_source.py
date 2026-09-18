"""Abstração fundamental de fonte de áudio para o Virtual Band AI.

Tanto arquivos de áudio (WAV/MP3) quanto fontes ao vivo (microfone/linha)
devem implementar esta interface. O restante do sistema (análise, contexto,
músicos virtuais) nunca precisa saber de onde veio o áudio.
"""

from abc import ABC, abstractmethod
import numpy as np


class AudioSource(ABC):
    """Interface abstrata para fontes de áudio."""

    @abstractmethod
    def read_chunk(self, chunk_size: int) -> np.ndarray:
        """Lê um bloco de áudio normalizado em ponto flutuante (mono, 1D float32 [-1.0, 1.0]).
        
        Utilizado principalmente pelo pipeline de análise espectral e harmônica (DSP).
        """
        pass

    @abstractmethod
    def read_playback_chunk(self, chunk_size: int) -> np.ndarray:
        """Lê um bloco de áudio com os canais originais (ex: estéreo float32) para reprodução audível."""
        pass

    @abstractmethod
    def get_sample_rate(self) -> int:
        """Retorna a taxa de amostragem em Hz (ex: 44100, 48000)."""
        pass

    @abstractmethod
    def get_duration(self) -> float:
        """Retorna a duração total do áudio em segundos (infinito ou 0 para transmissões ao vivo)."""
        pass

    @abstractmethod
    def get_position(self) -> float:
        """Retorna a posição atual do cursor de leitura em segundos."""
        pass

    @abstractmethod
    def seek(self, position_seconds: float) -> None:
        """Move o cursor de leitura para o tempo especificado (em segundos)."""
        pass

    @abstractmethod
    def is_active(self) -> bool:
        """Informa se a fonte de áudio está aberta e pronta para leitura."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Libera recursos abertos."""
        pass

    @property
    @abstractmethod
    def channels(self) -> int:
        """Retorna o número de canais de áudio da fonte (1=mono, 2=estéreo)."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Identificador textual ou nome do arquivo da fonte."""
        pass
