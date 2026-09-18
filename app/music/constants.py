"""Configurações centralizadas e ajustáveis para análise musical, estabilização e métricas.

Evita o espalhamento de 'números mágicos' pelo código, permitindo calibração fina
do equilíbrio entre estabilidade e velocidade de resposta do sistema.
"""
from typing import Optional

# ============================================================
# ESTABILIZAÇÃO DE ACORDES
# ============================================================
# Confiança mínima exigida para que uma detecção instantânea seja considerada
MIN_CHORD_CONFIDENCE: float = 0.40

# Tempo mínimo (em segundos) que um novo acorde deve persistir continuamente
# antes de iniciar o processo de transição (ex: 150 ms)
MIN_CHORD_STABILITY_TIME: float = 0.15

# Tempo contínuo (em segundos) de persistência para confirmar a mudança definitiva de acorde.
# Reduzido para 120 ms: com o cromagrama harmônico (mais limpo) podemos responder mais rápido
# às trocas de acorde sem introduzir falsos acordes — melhora a sensação de estar "no tempo".
CHORD_CHANGE_CONFIRMATION_TIME: float = 0.12

# Número máximo de eventos mantidos no histórico de acordes (None = ilimitado, armazena toda a faixa)
MAX_CHORD_HISTORY_ENTRIES: Optional[int] = None


# ============================================================
# ESTABILIZAÇÃO DE TONALIDADE (KEY)
# ============================================================
# Confiança mínima do coeficiente de correlação para aceitar uma tonalidade inicial
MIN_KEY_CONFIDENCE: float = 0.55

# Tempo mínimo (em segundos) que uma nova tonalidade candidata deve ser detectada
# continuamente superando a tonalidade titular para confirmar uma modulação harmônica real (ex: 5.0 s)
KEY_CHANGE_CONFIRMATION_TIME: float = 5.00

# Margem mínima de pontuação composta que a candidata precisa superar sobre a tonalidade atual
KEY_SCORE_MARGIN: float = 0.05

# Fatores de decaimento do acumulador de cromagrama
KEY_LEAKY_DECAY: float = 0.96          # Curto prazo (sensibilidade imediata a transientes)
KEY_LONG_TERM_DECAY: float = 0.990     # Longo prazo (~10 a 16 segundos de memória tonal global)

# Pesos da pontuação contextual composta de tonalidade
WEIGHT_KEY_INSTANTANEOUS: float = 0.20       # Sensibilidade imediata
WEIGHT_KEY_HISTORICAL_CHROMA: float = 0.40   # Distribuição de notas na janela de longo prazo
WEIGHT_KEY_CHORD_CONTEXT: float = 0.40       # Pertinência diatônica dos acordes recentes
WEIGHT_KEY_STABILITY: float = 0.10           # Inércia harmônica da tonalidade consolidada atual

# Número máximo de eventos no histórico de modulações (None = ilimitado, preserva toda a faixa)
MAX_KEY_HISTORY_ENTRIES: Optional[int] = None



# ============================================================
# RELÓGIO MUSICAL (MUSICAL CLOCK) & RITMO
# ============================================================
# Fórmula de compasso padrão
DEFAULT_METER: str = "4/4"
DEFAULT_BEATS_PER_BAR: int = 4
DEFAULT_BEAT_UNIT: int = 4
DEFAULT_BPM: float = 120.0

# Tolerância temporal para considerar um instante como pulso de batida (em segundos)
BEAT_PULSE_TOLERANCE_SEC: float = 0.080


# ============================================================
# LATÊNCIA E DSP
# ============================================================
# Tamanho padrão do bloco de análise de áudio (samples)
DEFAULT_CHUNK_SIZE: int = 4096

# Taxa de amostragem padrão
DEFAULT_SAMPLE_RATE: int = 44100


# ============================================================
# BAIXISTA VIRTUAL (BASSPLAYER v0.2)
# ============================================================
# Confiança harmônica mínima do acorde para que o baixo toque padrões complexos
MIN_BASS_CHORD_CONFIDENCE: float = 0.40

# Faixa de registro padrão do contrabaixo elétrico de 4 cordas em notas MIDI
# 28 = E1 (~41.2 Hz), 55 = G3 (~196.0 Hz)
BASS_MIN_MIDI_NOTE: int = 28
BASS_MAX_MIDI_NOTE: int = 55

# Oitava de referência padrão para a nota fundamental do baixo
BASS_DEFAULT_OCTAVE: int = 2

# Volume padrão do sintetizador de baixo (0.0 a 1.0)
BASS_DEFAULT_VOLUME: float = 0.85

# Razão de articulação da nota em relação à duração do beat (85% nota, 15% respiro)
BASS_NOTE_ARTICULATION_RATIO: float = 0.85


def midi_to_hz(midi: float) -> float:
    """Converte número de nota MIDI em frequência correspondente em Hertz (A4 = 440 Hz)."""
    return 440.0 * (2.0 ** ((float(midi) - 69.0) / 12.0))


