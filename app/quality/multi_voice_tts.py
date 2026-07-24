"""Multi-voice TTS manager — character voice assignment + emotion SSML.

Assigns distinct voices to each character in the storyboard and generates
SSML markup with emotion tags for Volcengine / Azure TTS.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class VoiceProfile:
    name: str
    voice_id: str  # volcengine / azure voice id
    gender: str = "female"
    age_range: str = "adult"
    default_emotion: str = "neutral"
    description: str = ""


# Built-in voice pool (Volcengine voices)
VOICE_POOL = [
    VoiceProfile("小欣", "zh_female_qingxin", "female", "young_adult", "neutral", "轻柔知性女声"),
    VoiceProfile("文侯", "zh_male_wenhou", "male", "adult", "neutral", "沉稳成熟男声"),
    VoiceProfile("甜莓", "zh_female_tianmei", "female", "young_adult", "happy", "活泼甜美女声"),
    VoiceProfile("精英", "zh_male_jingying", "male", "adult", "serious", "严肃精英男声"),
    VoiceProfile("少女", "zh_female_shaonv", "female", "teen", "shy", "羞涩少女音"),
    VoiceProfile("大叔", "zh_male_dashu", "male", "middle_age", "calm", "沉稳大叔音"),
]


class MultiVoiceTTS:
    """Manages multi-voice TTS for character dialogue.

    Assigns distinct voices to characters based on their profiles and
    generates SSML with emotion markup for expressive narration.

    Usage:
        mv = MultiVoiceTTS()
        mv.assign_voice("男主角", gender="male", age_range="adult", personality="冷静")
        ssml = mv.generate_ssml("我等你很久了。", character="男主角", emotion="serious")
    """

    def __init__(self):
        self._assignments: dict[str, VoiceProfile] = {}
        self._pool_index = 0

    def assign_voice(
        self,
        character_name: str,
        gender: str = "female",
        age_range: str = "adult",
        personality: str = "",
    ) -> VoiceProfile:
        """Assign a voice to a character. Returns the assigned profile.

        Uses a round-robin from matching pool entries; falls back to
        any available voice.
        """
        if character_name in self._assignments:
            return self._assignments[character_name]

        # Try to match by gender first
        candidates = [v for v in VOICE_POOL if v.gender == gender]
        if not candidates:
            candidates = VOICE_POOL

        profile = candidates[self._pool_index % len(candidates)]
        self._pool_index += 1
        self._assignments[character_name] = profile

        logger.info("Voice assigned: %s → %s (%s)", character_name, profile.name, profile.voice_id)
        return profile

    def get_voice(self, character_name: str) -> Optional[VoiceProfile]:
        return self._assignments.get(character_name)

    def generate_ssml(
        self,
        text: str,
        character: str = "",
        emotion: str = "neutral",
        speed: float = 1.0,
        narrator: bool = False,
    ) -> str:
        """Generate SSML for a line of dialogue with emotion markup.

        Args:
            text: The spoken text
            character: Character name (for voice assignment)
            emotion: neutral / happy / sad / angry / fearful / surprised / serious
            speed: Speech rate multiplier
            narrator: If True, use narrator voice instead of character voice
        """
        if narrator or not character:
            voice_id = "zh_female_qingxin"
        else:
            profile = self._assignments.get(character)
            voice_id = profile.voice_id if profile else "zh_female_qingxin"

        emotion_map = {
            "neutral": "neutral",
            "happy": "cheerful", "sad": "sad", "angry": "angry",
            "fearful": "fearful", "surprised": "excited", "serious": "serious",
        }
        emotion_tag = emotion_map.get(emotion, "neutral")

        ssml = (
            f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
            f'xmlns:mstts="http://www.w3.org/2001/mstts" xml:lang="zh-CN">'
            f'<voice name="{voice_id}">'
            f'<mstts:express-as style="{emotion_tag}" styledegree="1.0">'
            f'<prosody rate="{speed}">{text}</prosody>'
            f"</mstts:express-as>"
            f"</voice></speak>"
        )
        return ssml

    def generate_narration_script(
        self,
        lines: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Generate a full TTS script with voice assignments for all lines.

        Each line: {"text": "...", "character": "", "emotion": "neutral", "type": "dialogue|narration"}
        Returns enriched lines with voice_id and ssml.
        """
        result = []
        for line in lines:
            character = line.get("character", "")
            emotion = line.get("emotion", "neutral")
            text = line.get("text", "")
            line_type = line.get("type", "dialogue")

            is_narrator = line_type == "narration" or not character
            voice_id = "zh_female_qingxin" if is_narrator else (
                self._assignments.get(character, VOICE_POOL[0]).voice_id
            )

            result.append({
                **line,
                "voice_id": voice_id,
                "ssml": self.generate_ssml(text, character, emotion, narrator=is_narrator),
            })
        return result

    def list_assignments(self) -> dict[str, dict]:
        return {
            name: {"voice_name": p.name, "voice_id": p.voice_id, "gender": p.gender}
            for name, p in self._assignments.items()
        }


# Module-level singleton
multi_voice_tts = MultiVoiceTTS()
