"""BGM selector — selects background music based on scene emotion and genre.

Matches scene emotion to a curated BGM library. For demo purposes, provides
a mapping of emotion categories to BGM presets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class BGMTrack:
    name: str
    path: str
    duration_seconds: float
    emotion: str
    intensity: str = "medium"  # low / medium / high
    genre: str = "general"
    bpm: int = 120


# Curated BGM library (emotion → track mapping)
BGM_LIBRARY: dict[str, list[BGMTrack]] = {
    "happy": [
        BGMTrack("轻快日常", "bgm/happy_daily.mp3", 120, "happy", "low", "slice_of_life", 110),
        BGMTrack("欢快冒险", "bgm/happy_adventure.mp3", 90, "happy", "medium", "adventure", 130),
    ],
    "sad": [
        BGMTrack("忧伤叙事", "bgm/sad_narrative.mp3", 150, "sad", "low", "drama", 70),
        BGMTrack("深情回忆", "bgm/sad_memory.mp3", 100, "sad", "medium", "romance", 65),
    ],
    "tense": [
        BGMTrack("悬疑紧张", "bgm/tense_suspense.mp3", 80, "tense", "medium", "thriller", 100),
        BGMTrack("追逐战斗", "bgm/tense_chase.mp3", 60, "tense", "high", "action", 160),
    ],
    "epic": [
        BGMTrack("史诗决战", "bgm/epic_battle.mp3", 120, "epic", "high", "action", 140),
        BGMTrack("命运转折", "bgm/epic_turning.mp3", 90, "epic", "medium", "drama", 85),
    ],
    "romantic": [
        BGMTrack("温馨相遇", "bgm/romantic_meet.mp3", 100, "romantic", "low", "romance", 80),
        BGMTrack("深情告白", "bgm/romantic_confess.mp3", 80, "romantic", "medium", "romance", 75),
    ],
    "mysterious": [
        BGMTrack("神秘开场", "bgm/mysterious_open.mp3", 60, "mysterious", "low", "mystery", 90),
        BGMTrack("揭晓真相", "bgm/mysterious_reveal.mp3", 70, "mysterious", "high", "mystery", 100),
    ],
    "neutral": [
        BGMTrack("日常过渡", "bgm/neutral_transition.mp3", 30, "neutral", "low", "general", 100),
    ],
}


class BGMSelector:
    """Selects BGM tracks based on scene emotion and intensity.

    Usage:
        selector = BGMSelector()
        track = selector.select(emotion="happy", intensity="medium")
    """

    def __init__(self):
        self._bgm_library = BGM_LIBRARY.copy()

    def select(
        self,
        emotion: str = "neutral",
        intensity: str = "medium",
        genre: str = "general",
    ) -> Optional[BGMTrack]:
        """Select the best matching BGM track.

        Priority: exact emotion match → intensity match → genre match → neutral fallback.
        """
        candidates = self._bgm_library.get(emotion, [])
        if not candidates:
            candidates = self._bgm_library.get("neutral", [])

        # Filter by intensity
        intensity_matches = [t for t in candidates if t.intensity == intensity]
        if not intensity_matches:
            intensity_matches = candidates

        # Filter by genre
        genre_matches = [t for t in intensity_matches if t.genre == genre]
        if not genre_matches:
            genre_matches = intensity_matches

        selected = genre_matches[0] if genre_matches else None
        if selected:
            logger.debug("BGM selected: %s (emotion=%s, intensity=%s)", selected.name, emotion, intensity)
        return selected

    def select_for_scene(
        self,
        scene: dict[str, Any],
    ) -> Optional[BGMTrack]:
        """Select BGM for a scene based on its metadata.

        Scene dict keys: emotion, intensity, genre, is_climax, is_opening, is_ending
        """
        emotion = scene.get("emotion", "neutral")
        intensity = scene.get("intensity", "medium")
        genre = scene.get("genre", "general")

        # Climax scenes get epic BGM
        if scene.get("is_climax"):
            emotion = "epic"
            intensity = "high"

        # Opening / ending get special treatment
        if scene.get("is_opening"):
            emotion = "epic"
        elif scene.get("is_ending"):
            emotion = "sad" if scene.get("ending_tone") == "bittersweet" else "romantic"

        return self.select(emotion=emotion, intensity=intensity, genre=genre)

    def add_custom_track(self, track: BGMTrack) -> None:
        """Add a custom BGM track to the library."""
        if track.emotion not in self._bgm_library:
            self._bgm_library[track.emotion] = []
        self._bgm_library[track.emotion].append(track)


# Module-level singleton
bgm_selector = BGMSelector()
