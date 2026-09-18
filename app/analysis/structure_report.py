"""Exportador de Relatórios Estruturais em JSON (v0.3).

Gera relatórios estruturados para debugging, auditoria e futura comunicação
com outros agentes ou músicos virtuais da banda.
"""

import json
import os
from typing import Dict, Any, Optional

from app.music.music_structure import MusicStructure
from app.music.prediction_engine import MusicPrediction


def export_structure_report(
    structure: MusicStructure,
    output_path_or_duration: Any = "music_structure_report.json",
    total_duration: float = 0.0,
    bpm: float = 120.0,
    key: str = "C Major",
    output_path: Optional[str] = None,
    prediction: Optional[MusicPrediction] = None
) -> str:
    """Gera um arquivo JSON formatado com a análise estrutural completa da composição."""
    if isinstance(output_path_or_duration, str):
        target_output_path = output_path_or_duration
    else:
        target_output_path = output_path if output_path is not None else "music_structure_report.json"
        total_duration = float(output_path_or_duration)

    struct_dict = structure.to_dict()
    sections = struct_dict.get("sections", [])
    patterns = struct_dict.get("patterns", {})
    transitions = struct_dict.get("transitions", [])

    if total_duration <= 0.0 and sections:
        total_duration = max(float(s.get("end_time", 0.0)) for s in sections)

    abstract_seq = struct_dict.get("abstract_sequence", [])
    struct_seq = struct_dict.get("structure_sequence", [])
    conf = struct_dict.get("analysis_confidence", 0.0)

    report: Dict[str, Any] = {
        "song_summary": {
            "duration_seconds": round(total_duration, 2),
            "bpm": round(bpm, 1),
            "key": key,
            "total_sections": len(sections),
            "total_patterns": len(patterns) if isinstance(patterns, (list, dict)) else 0
        },
        "sections": sections,
        "patterns": patterns,
        "transitions": transitions,
        "musical_form": {
            "structure_sequence": struct_seq,
            "abstract_sequence": abstract_seq,
            "form_string": " - ".join(abstract_seq) if abstract_seq else "--"
        },
        "analysis_metadata": {
            "version": "v0.3",
            "analysis_confidence": round(conf, 2),
            "system": "Virtual Band AI - Music Structure Analyzer"
        },
        # Campos de conveniência no topo
        "version": "v0.3",
        "duration_seconds": round(total_duration, 2),
        "bpm": round(bpm, 1),
        "key": key,
        "analysis_confidence": round(conf, 2),
        "structure_sequence": struct_seq,
        "abstract_sequence": abstract_seq,
        "current_position": struct_dict.get("current_position", {}),
        "predictions": [prediction.to_dict()] if prediction else []
    }

    abs_path = os.path.abspath(target_output_path)
    with open(abs_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    return abs_path

