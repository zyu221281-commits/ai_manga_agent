"""Style template manager — visual style presets for consistent aesthetic.

Manages named style templates (国风/日系/写实/水墨/赛博朋克) with
consistent prompt prefixes for both image generation and video compositing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class StyleTemplate:
    name: str
    display_name: str
    image_prompt_prefix: str
    image_negative_prompt: str
    color_palette: list[str]
    subtitle_font: str = "Microsoft YaHei"
    subtitle_color: str = "&H00FFFFFF"
    cover_style: str = "default"


# Built-in style templates
STYLE_TEMPLATES: dict[str, StyleTemplate] = {
    "guofeng": StyleTemplate(
        name="guofeng",
        display_name="国风古装",
        image_prompt_prefix=(
            "Chinese ink painting style, traditional Chinese aesthetics, "
            "flowing brushstrokes, elegant ancient Chinese attire, "
            "guzheng-inspired atmosphere, soft natural lighting"
        ),
        image_negative_prompt=(
            "western style, modern clothing, 3D render, photorealistic, "
            "text, watermark, signature, deformed, ugly"
        ),
        color_palette=["#C23A2B", "#2B3A67", "#D4A574", "#F5E6D3", "#1A1A2E"],
        subtitle_font="Noto Serif CJK SC",
        subtitle_color="&H00E8D5B7",
        cover_style="国风漫剧",
    ),
    "rihan": StyleTemplate(
        name="rihan",
        display_name="日系动漫",
        image_prompt_prefix=(
            "Japanese anime style, cel-shaded, vibrant colors, "
            "clean lineart, anime screencap, studio quality animation"
        ),
        image_negative_prompt=(
            "3D render, realistic, photorealistic, western cartoon style, "
            "text, watermark, deformed hands, bad anatomy"
        ),
        color_palette=["#FF6B6B", "#4ECDC4", "#FFE66D", "#1A535C", "#F7FFF7"],
        subtitle_font="Microsoft YaHei",
        subtitle_color="&H00FFFFFF",
        cover_style="日系动漫",
    ),
    "xieshi": StyleTemplate(
        name="xieshi",
        display_name="写实风格",
        image_prompt_prefix=(
            "Semi-realistic Chinese illustration, detailed facial features, "
            "dramatic lighting, cinematic composition, high quality render"
        ),
        image_negative_prompt=(
            "cartoon, anime, deformed, ugly, blurry, low quality, "
            "text, watermark, oversaturated"
        ),
        color_palette=["#2D3436", "#636E72", "#DFE6E9", "#B2BEC3", "#74B9FF"],
        subtitle_font="Microsoft YaHei",
        subtitle_color="&H00EAEAEA",
        cover_style="写实漫剧",
    ),
    "shuimo": StyleTemplate(
        name="shuimo",
        display_name="水墨丹青",
        image_prompt_prefix=(
            "Traditional Chinese ink wash painting, brush and ink, "
            "monochrome with subtle color accents, poetic atmosphere, "
            "Song dynasty landscape aesthetics"
        ),
        image_negative_prompt=(
            "vibrant colors, western style, 3D, photorealistic, "
            "text, watermark, cartoon, anime"
        ),
        color_palette=["#1A1A1A", "#4A4A4A", "#8B8B8B", "#D4D4D4", "#C23A2B"],
        subtitle_font="Noto Serif CJK SC",
        subtitle_color="&H00D4D4D4",
        cover_style="水墨漫剧",
    ),
    "saibo": StyleTemplate(
        name="saibo",
        display_name="赛博朋克",
        image_prompt_prefix=(
            "Cyberpunk Chinese aesthetics, neon-lit cityscape, "
            "high-tech low-life, holographic interfaces, purple and cyan lighting"
        ),
        image_negative_prompt=(
            "rural, natural landscape, ancient, historical, "
            "text, watermark, low quality"
        ),
        color_palette=["#FF006E", "#8338EC", "#3A86FF", "#00F5D4", "#FFBE0B"],
        subtitle_font="Microsoft YaHei",
        subtitle_color="&H0000F5D4",
        cover_style="赛博漫剧",
    ),
}


class StyleTemplateManager:
    """Manages visual style templates for the pipeline.

    Usage:
        mgr = StyleTemplateManager()
        template = mgr.get("guofeng")
        prompt = template.image_prompt_prefix + "a warrior standing on a cliff"
    """

    def __init__(self):
        self._templates = STYLE_TEMPLATES.copy()

    def get(self, name: str) -> Optional[StyleTemplate]:
        """Get a style template by name."""
        return self._templates.get(name)

    def get_image_prompt(self, style_name: str, base_prompt: str) -> str:
        """Build a full image prompt with style prefix."""
        template = self.get(style_name)
        if template is None:
            return base_prompt
        return f"{template.image_prompt_prefix}, {base_prompt}"

    def get_negative_prompt(self, style_name: str) -> str:
        """Get the negative prompt for a style."""
        template = self.get(style_name)
        if template is None:
            return "text, watermark, deformed, low quality"
        return template.image_negative_prompt

    def list_styles(self) -> list[dict]:
        """List all available styles."""
        return [
            {
                "name": t.name,
                "display_name": t.display_name,
                "palette": t.color_palette[:3],
            }
            for t in self._templates.values()
        ]

    def register_template(self, template: StyleTemplate) -> None:
        """Register a custom style template."""
        self._templates[template.name] = template
        logger.info("Style template registered: %s (%s)", template.name, template.display_name)

    def get_for_genre(self, genre: str) -> str:
        """Recommend a style based on genre."""
        genre_map = {
            "仙侠": "guofeng",
            "玄幻": "guofeng",
            "都市": "xieshi",
            "言情": "rihan",
            "悬疑": "xieshi",
            "科幻": "saibo",
            "古言": "shuimo",
        }
        return genre_map.get(genre, "rihan")


# Module-level singleton
style_template = StyleTemplateManager()
