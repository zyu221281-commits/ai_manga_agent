"""Subtitle generator — SRT/ASS generation from TTS word timestamps.

Generates timed subtitle files (.srt / .ass) from TTS word-level timestamps
or from script text with estimated duration. Supports Chinese and English.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SubtitleLine:
    index: int
    start_ms: int
    end_ms: int
    text: str
    speaker: str = ""


class SubtitleGenerator:
    """Generates SRT/ASS subtitle files for video compositing.

    Input: script lines with optional word-level timestamps from Azure TTS.
    Output: SRT file path ready for FFmpeg subtitle overlay.

    Usage:
        gen = SubtitleGenerator()
        srt_path = gen.generate_srt(script_lines, output_path="ep_001.srt")
    """

    LINE_MAX_CHARS = 20  # Max characters per subtitle line
    LINE_MIN_DURATION_MS = 1500  # Minimum duration per subtitle
    CHARS_PER_SECOND = 4.5  # Average reading speed for Chinese

    def __init__(
        self,
        max_chars: int = LINE_MAX_CHARS,
        min_duration_ms: int = LINE_MIN_DURATION_MS,
    ):
        self._max_chars = max_chars
        self._min_duration_ms = min_duration_ms

    def generate_srt(
        self,
        lines: list[dict[str, Any]],
        output_path: str,
        base_start_ms: int = 0,
    ) -> str:
        """Generate an SRT file from script/dialogue lines.

        Args:
            lines: List of {"text": str, "speaker": str, "emotion": str, ...}
            output_path: Output .srt file path
            base_start_ms: Starting time offset in ms

        Returns:
            The output file path.
        """
        subtitle_lines = self._text_to_subtitle_lines(lines, base_start_ms)
        srt_content = self._format_srt(subtitle_lines)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        logger.info("SRT generated: %d lines → %s", len(subtitle_lines), output_path)
        return output_path

    def _text_to_subtitle_lines(
        self,
        lines: list[dict[str, Any]],
        base_ms: int,
    ) -> list[SubtitleLine]:
        """Convert narration lines to timed subtitle lines."""
        result = []
        current_ms = base_ms
        index = 1

        for line in lines:
            text = line.get("text", "")
            speaker = line.get("speaker", "")

            # Estimate duration based on character count
            char_count = len(text)
            estimated_duration_ms = max(
                self._min_duration_ms,
                int(char_count / self.CHARS_PER_SECOND * 1000),
            )

            # Split long lines
            if char_count > self._max_chars:
                chunks = self._split_text(text)
                for chunk in chunks:
                    chunk_duration = max(
                        self._min_duration_ms,
                        int(len(chunk) / self.CHARS_PER_SECOND * 1000),
                    )
                    result.append(SubtitleLine(
                        index=index,
                        start_ms=current_ms,
                        end_ms=current_ms + chunk_duration,
                        text=chunk,
                        speaker=speaker,
                    ))
                    current_ms += chunk_duration
                    index += 1
            else:
                result.append(SubtitleLine(
                    index=index,
                    start_ms=current_ms,
                    end_ms=current_ms + estimated_duration_ms,
                    text=text,
                    speaker=speaker,
                ))
                current_ms += estimated_duration_ms
                index += 1

        return result

    @staticmethod
    def _split_text(text: str, max_chars: int = 20) -> list[str]:
        """Split long text into subtitle-sized chunks at natural breaks."""
        chunks = []
        remaining = text
        while len(remaining) > max_chars:
            # Try to split at punctuation
            split_at = max_chars
            for punct in ["。", "，", "、", "；", "？", "！", " ", ",", "."]:
                pos = remaining[:max_chars].rfind(punct)
                if pos > 0:
                    split_at = pos + 1
                    break
            chunks.append(remaining[:split_at].strip())
            remaining = remaining[split_at:].strip()
        if remaining:
            chunks.append(remaining)
        return chunks

    @staticmethod
    def _format_srt(lines: list[SubtitleLine]) -> str:
        """Format subtitle lines as SRT text."""
        parts = []
        for line in lines:
            start_ts = _ms_to_timestamp(line.start_ms)
            end_ts = _ms_to_timestamp(line.end_ms)
            prefix = f"{line.speaker}: " if line.speaker else ""
            parts.append(f"{line.index}\n{start_ts} --> {end_ts}\n{prefix}{line.text}\n")
        return "\n".join(parts)

    def generate_ass(
        self,
        lines: list[dict[str, Any]],
        output_path: str,
        style: str = "Default",
        font_size: int = 18,
    ) -> str:
        """Generate an ASS (Advanced SubStation Alpha) file with styling.

        ASS supports font styles, colors, and positioning beyond SRT.
        """
        header = (
            "[Script Info]\n"
            "Title: AI Manga Agent Subtitles\n"
            "ScriptType: .00+\n"
            "WrapStyle: 2\n"
            "PlayResX: 1080\n"
            "PlayResY: 1920\n\n"
            "[+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding\n"
            f"Style: Default,Microsoft YaHei,{font_size},&H00FFFFFF,&H000000FF,"
            f"&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,60,60,60,1\n\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        )

        subtitle_lines = self._text_to_subtitle_lines(lines, 0)
        events = []
        for line in subtitle_lines:
            start_ts = _ms_to_ass_timestamp(line.start_ms)
            end_ts = _ms_to_ass_timestamp(line.end_ms)
            prefix = f"{line.speaker}: " if line.speaker else ""
            events.append(
                f"Dialogue: 0,{start_ts},{end_ts},Default,,0,0,0,,{prefix}{line.text}"
            )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(header + "\n".join(events) + "\n")

        logger.info("ASS generated: %d lines → %s", len(lines), output_path)
        return output_path


def _ms_to_timestamp(ms: int) -> str:
    """Convert milliseconds to SRT timestamp: HH:MM:SS,mmm"""
    hours = ms // 3600000
    mins = (ms % 3600000) // 60000
    secs = (ms % 60000) // 1000
    millis = ms % 1000
    return f"{hours:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def _ms_to_ass_timestamp(ms: int) -> str:
    """Convert milliseconds to ASS timestamp: H:MM:SS.cc"""
    hours = ms // 3600000
    mins = (ms % 3600000) // 60000
    secs = (ms % 60000) // 1000
    centis = (ms % 1000) // 10
    return f"{hours}:{mins:02d}:{secs:02d}.{centis:02d}"
