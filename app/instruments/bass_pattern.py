"""Gerador de Padrões Rítmicos do Baixista Virtual (BassPatternGenerator).

Responsável por estruturar a sequência de notas a serem tocadas a cada tempo do compasso:
- ROOT: Fundamental nos tempos principais (ex: C C C C)
- ROOT_FIFTH: Alternância clássica de fundamental e quinta (ex: C G C G)
- ROOT_FIFTH_OCTAVE: Fundamental, quinta e oitava (ex: C G C3 G)
- SUSTAINED: Fundamental longa sustentada
- AUTO: Seleção inteligente baseada na duração do acorde e na confiança harmônica
"""

from typing import List, Tuple
from app.instruments.bass_model import BassDecision, BassPatternType, BassNoteValue
from app.music.constants import MIN_BASS_CHORD_CONFIDENCE


class BassPatternGenerator:
    """Gera o padrão rítmico de notas por tempo do compasso."""

    def __init__(self, min_confidence: float = MIN_BASS_CHORD_CONFIDENCE):
        self._min_confidence = min_confidence

    def generate_pattern(
        self,
        decision: BassDecision,
        beats: int = 4,
        chord_duration_beats: float = 4.0,
        note_value: BassNoteValue = BassNoteValue.QUARTER,
    ) -> List[Tuple[float, str, int, str]]:
        """Gera lista de tuplas (beat, note_name, midi_note, reason) para o compasso/acorde.
        
        Exemplo de retorno para C em ROOT_FIFTH (4 beats):
          [(1, 'C2', 36, 'chord_root'),
           (2, 'G1', 31, 'chord_fifth'),
           (3, 'C2', 36, 'chord_root'),
           (4, 'G1', 31, 'chord_fifth')]
        """
        pat_type = decision.pattern_type

        # Resolução do modo AUTO
        if pat_type == BassPatternType.AUTO:
            if decision.confidence < self._min_confidence:
                # Baixa confiança: simplifica para apenas fundamental (segurança harmônica)
                pat_type = BassPatternType.ROOT
            elif chord_duration_beats >= 3.5:
                # Acorde completo de 4 tempos: padrão clássico Fundamental + Quinta
                pat_type = BassPatternType.ROOT_FIFTH
            elif chord_duration_beats >= 1.5:
                pat_type = BassPatternType.ROOT_FIFTH
            else:
                pat_type = BassPatternType.ROOT

        effective_beats = min(beats, max(1, int(round(chord_duration_beats))))

        # -------------------------------------------------------------
        # 1. Padrão ROOT (Apenas Fundamental)
        # -------------------------------------------------------------
        if pat_type in (BassPatternType.ROOT, BassPatternType.FUNDAMENTALS):
            events = []
            position = 1.0
            limit = 1.0 + min(float(beats), max(0.0, chord_duration_beats))
            while position < limit - 1e-9:
                events.append((position, decision.root_note, decision.root_midi, "Fundamental"))
                position += note_value.beats
            return events

        # -------------------------------------------------------------
        # 2. Padrão ROOT + FIFTH (Fundamental e Quinta Alternadas)
        # -------------------------------------------------------------
        elif pat_type == BassPatternType.ROOT_FIFTH:
            events = []
            for b in range(1, effective_beats + 1):
                if b % 2 == 1:
                    events.append((b, decision.root_note, decision.root_midi, "Fundamental"))
                else:
                    events.append((b, decision.fifth_note, decision.fifth_midi, "Quinta Justa"))
            return events

        # -------------------------------------------------------------
        # 3. Padrão ROOT + FIFTH + OCTAVE
        # -------------------------------------------------------------
        elif pat_type == BassPatternType.ROOT_FIFTH_OCTAVE:
            events = []
            for b in range(1, effective_beats + 1):
                if b == 1:
                    events.append((b, decision.root_note, decision.root_midi, "Fundamental"))
                elif b == 2:
                    events.append((b, decision.fifth_note, decision.fifth_midi, "Quinta Justa"))
                elif b == 3:
                    events.append((b, decision.octave_note, decision.octave_midi, "Oitava"))
                else:
                    events.append((b, decision.fifth_note, decision.fifth_midi, "Quinta Justa"))
            return events

        # -------------------------------------------------------------
        # 4. Padrão SUSTAINED (Nota Única Sustentada)
        # -------------------------------------------------------------
        elif pat_type == BassPatternType.SUSTAINED:
            return [(1, decision.root_note, decision.root_midi, "Fundamental (Sustentada)")]

        # Fallback para ROOT
        return [(b, decision.root_note, decision.root_midi, "Fundamental") for b in range(1, effective_beats + 1)]
