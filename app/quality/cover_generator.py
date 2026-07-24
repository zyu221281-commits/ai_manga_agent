"""Cover generator — title card and thumbnail generation for episodes.

Generates episode cover images with title, episode number, and style
consistent with the series visual identity. Can also generate platform-
specific thumbnails (16:9 for B站/YouTube, 3:4 for 抖音).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CoverSpec:
    title: str
    episode_num: int
    subtitle: str = ""
    style: str = "default"
    style_template: str = "vertical_short"
    target_platform: str = "douyin"
    width: int = 1080
    height: int = 1920


@dataclass
class CoverResult:
    success: bool
    image_url: Optional[str] = None
    image_bytes: Optional[bytes] = None
    width: int = 1080
    height: int = 1920
    error: Optional[str] = None


PLATFORM_SIZES = {
    "douyin": (1080, 1920),    # 9:16 vertical short
    "kuaishou": (1080, 1920),  # 9:16 vertical short
    "bilibili": (1920, 1080),  # 16:9 horizontal
    "youtube": (1280, 720),    # 16:9 thumbnail
}


class CoverGenerator:
    """Generates cover images and thumbnails for each episode.

    Uses cloud image generation (Flux) with a structured prompt that
    includes the episode title, number, and visual style.

    Usage:
        gen = CoverGenerator()
        result = await gen.generate(CoverSpec(title="觉醒", episode_num=1))
    """

    def __init__(self, default_style: str = "国风漫剧"):
        self._default_style = default_style

    async def generate(
        self,
        spec: CoverSpec,
    ) -> CoverResult:
        """Generate a cover image for an episode.

        Builds a structured prompt from the cover spec and calls
        the cloud image generator.
        """
        prompt = self._build_prompt(spec)

        try:
            from app.services.image_gen_cloud import cloud_image_gen

            size = PLATFORM_SIZES.get(spec.target_platform, (1080, 1920))
            gen_result = await cloud_image_gen.generate(
                prompt=prompt,
                model="flux-schnell",
                width=size[0],
                height=size[1],
                num_images=1,
            )

            if gen_result.success:
                logger.info("Cover generated: Ep %d: %s", spec.episode_num, spec.title)
                return CoverResult(
                    success=True,
                    image_url=gen_result.image_url,
                    width=size[0],
                    height=size[1],
                )
            else:
                return CoverResult(success=False, error=gen_result.error)
        except Exception as e:
            logger.error("Cover generation failed for Ep %d: %s", spec.episode_num, str(e)[:200])
            return CoverResult(success=False, error=str(e))

    def _build_prompt(self, spec: CoverSpec) -> str:
        """Build an image generation prompt for the cover."""
        parts = [
            "Anime manga cover illustration",
            f"Title: {spec.title}",
            f"Episode {spec.episode_num}",
        ]
        if spec.subtitle:
            parts.append(f"Subtitle: {spec.subtitle}")

        parts.extend([
            f"Style: {spec.style or self._default_style}",
            "Vertical composition, dramatic lighting, professional anime cover art",
            "Clean typography area for text overlay, cinematic atmosphere",
            "No watermark",
        ])
        return ", ".join(parts)

    def generate_thumbnail(
        self,
        spec: CoverSpec,
    ) -> CoverResult:
        """Generate a platform thumbnail variant (e.g., 16:9 for YouTube/B站)."""
        # Resize for horizontal platforms
        if spec.target_platform in ("bilibili", "youtube"):
            size = PLATFORM_SIZES.get(spec.target_platform, (1920, 1080))
            spec.width = size[0]
            spec.height = size[1]
        return self.generate(spec)


# Module-level singleton
cover_generator = CoverGenerator()
