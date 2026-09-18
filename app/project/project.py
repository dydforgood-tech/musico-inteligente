"""Modelo do Projeto Musical Geral (Project v0.4).

Raiz da hierarquia do Virtual Band AI contendo múltiplos Setlists, configurações
globais de áudio e metadados, persistido 100% localmente em JSON.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
import uuid

from app.setlist.setlist import Setlist
from app.song.song import Song


@dataclass
class ProjectSettings:
    """Configurações globais de operação do projeto."""
    sample_rate: int = 44100
    chunk_size: int = 4096
    auto_save: bool = True
    auto_save_interval_sec: int = 60
    follow_mode_default: bool = True
    audio_buffer_size: int = 4096
    theme: str = "dark"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "chunk_size": self.chunk_size,
            "auto_save": self.auto_save,
            "auto_save_interval_sec": self.auto_save_interval_sec,
            "follow_mode_default": self.follow_mode_default,
            "audio_buffer_size": self.audio_buffer_size,
            "theme": self.theme,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ProjectSettings":
        if not data:
            return cls()
        return cls(
            sample_rate=int(data.get("sample_rate", 44100)),
            chunk_size=int(data.get("chunk_size", 4096)),
            auto_save=bool(data.get("auto_save", True)),
            auto_save_interval_sec=int(data.get("auto_save_interval_sec", 60)),
            follow_mode_default=bool(data.get("follow_mode_default", True)),
            audio_buffer_size=int(data.get("audio_buffer_size", 4096)),
            theme=str(data.get("theme", "dark")),
        )


@dataclass
class Project:
    """Modelo raiz que encapsula todo o repertório, setlists e configurações da banda."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = "Novo Projeto de Banda"
    description: str = "Repertório Musical e Play-Along"
    version: str = "v0.4"
    setlists: List[Setlist] = field(default_factory=list)
    active_setlist_id: Optional[str] = None
    settings: ProjectSettings = field(default_factory=ProjectSettings)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self):
        if not self.setlists:
            initial_set = Setlist(name="Repertório Principal")
            self.setlists.append(initial_set)
            self.active_setlist_id = initial_set.id

    def get_active_setlist(self) -> Setlist:
        for s in self.setlists:
            if s.id == self.active_setlist_id:
                return s
        if self.setlists:
            self.active_setlist_id = self.setlists[0].id
            return self.setlists[0]
        new_set = Setlist(name="Repertório Principal")
        self.setlists.append(new_set)
        self.active_setlist_id = new_set.id
        return new_set

    def get_song_by_id(self, song_id: str) -> Optional[Song]:
        """Busca uma música pelo ID em todos os setlists do projeto."""
        for s in self.setlists:
            song = s.get_song(song_id)
            if song:
                return song
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "active_setlist_id": self.active_setlist_id,
            "setlists": [s.to_dict() for s in self.setlists],
            "settings": self.settings.to_dict(),
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Project":
        set_raw = data.get("setlists", [])
        settings_dict = data.get("settings", {})
        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            name=data.get("name", "Novo Projeto de Banda"),
            description=data.get("description", ""),
            version=data.get("version", "v0.4"),
            active_setlist_id=data.get("active_setlist_id"),
            setlists=[Setlist.from_dict(s) for s in set_raw],
            settings=ProjectSettings.from_dict(settings_dict),
            metadata=dict(data.get("metadata", {})),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
        )
