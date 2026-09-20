"""Confiança consolidada para decidir quanta iniciativa a banda pode tomar."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FollowConfidence:
    """Resultado explicável; não substitui as confianças de cada detector."""
    score: float
    level: str


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def calculate_follow_confidence(*, tempo: float, phase: float, position: float,
                                harmonic: float, chart_alignment: float,
                                stability: float) -> FollowConfidence:
    """Combina evidências independentes de forma conservadora.

    Tempo/fase, posição e referência da cifra são grupos diferentes. A menor
    evidência de alinhamento limita o grupo para que uma média alta não esconda
    uma posição ou uma cifra insegura.
    """
    tempo = _clamp(tempo)
    phase = _clamp(phase)
    position = _clamp(position)
    harmonic = _clamp(harmonic)
    chart_alignment = _clamp(chart_alignment)
    stability = _clamp(stability)

    temporal = 0.60 * tempo + 0.40 * phase
    alignment = min(position, chart_alignment)
    score = 0.42 * temporal + 0.32 * alignment + 0.18 * harmonic + 0.08 * stability

    # Um bom resultado em um detector não compensa a ausência de base temporal
    # ou de localização. São gates, não uma segunda média de confiança.
    if temporal < 0.35:
        score *= 0.65
    if alignment < 0.35:
        score *= 0.70
    if stability < 0.30:
        score *= 0.80
    score = _clamp(score)

    if score >= 0.75 and min(temporal, alignment, harmonic, stability) >= 0.55:
        level = "HIGH"
    elif score >= 0.45 and alignment >= 0.35:
        level = "MEDIUM"
    else:
        level = "LOW"
    return FollowConfidence(score=score, level=level)
