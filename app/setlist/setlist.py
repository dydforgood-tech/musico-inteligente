"""Modelo de Repertório/Setlist (Setlist v0.4).

Agrupa e ordena músicas em uma lista de apresentação executável pela banda.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import uuid

from app.song.song import Song


@dataclass
class Setlist:
    """Representação de um repertório ou lista de músicas (Setlist)."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = "Novo Repertório"
    description: str = ""
    songs: List[Song] = field(default_factory=list)
    active_song_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_song(self, song: Song, index: Optional[int] = None) -> None:
        """Adiciona uma música ao setlist em posição específica ou no final."""
        if index is not None and 0 <= index <= len(self.songs):
            self.songs.insert(index, song)
        else:
            self.songs.append(song)
        if not self.active_song_id:
            self.active_song_id = song.id

    def remove_song(self, song_id: str) -> bool:
        """Remove uma música pelo ID."""
        initial_len = len(self.songs)
        self.songs = [s for s in self.songs if s.id != song_id]
        if self.active_song_id == song_id:
            self.active_song_id = self.songs[0].id if self.songs else None
        return len(self.songs) < initial_len

    def get_song(self, song_id: str) -> Optional[Song]:
        for s in self.songs:
            if s.id == song_id:
                return s
        return None

    def get_active_song(self) -> Optional[Song]:
        if not self.active_song_id and self.songs:
            self.active_song_id = self.songs[0].id
        return self.get_song(self.active_song_id) if self.active_song_id else None

    def reorder(self, song_ids_order: List[str]) -> None:
        """Reordena a lista de músicas de acordo com a lista de IDs fornecida."""
        lookup = {s.id: s for s in self.songs}
        reordered = []
        for sid in song_ids_order:
            if sid in lookup:
                reordered.append(lookup[sid])
        # Inclui quaisquer remanescentes que não constavam na lista
        for s in self.songs:
            if s not in reordered:
                reordered.append(s)
        self.songs = reordered

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "active_song_id": self.active_song_id,
            "songs": [s.to_dict() for s in self.songs],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Setlist":
        songs_raw = data.get("songs", [])
        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            name=data.get("name", "Novo Repertório"),
            description=data.get("description", ""),
            active_song_id=data.get("active_song_id"),
            songs=[Song.from_dict(s) for s in songs_raw],
            metadata=dict(data.get("metadata", {})),
        )
