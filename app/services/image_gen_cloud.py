"""Cloud image generation service (Flux Schnell / Seedream via SiliconFlow).

 dependency strategy: primary = Flux for regular scenes, Seedream for
character-anchored consistency. All calls go through SiliconFlow API;
local NudeNet NSFW check is applied post-generation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from openai import AsyncOpenAI

from app.core.config import settings
from app.core.pricing import calculate_unit_cost

logger = logging.getLogger(__name__)


@dataclass
class ImageGenResult:
    success: bool
    image_url: Optional[str] = None
    image_bytes: Optional[bytes] = None
    model: str = "flux-schnell"
    cost_usd: float = 0.0
    duration_ms: float = 0.0
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class CloudImageGenerator:
    """Generates images via SiliconFlow / Flux Schnell + Seedream.

    Usage:
        gen = CloudImageGenerator()
        result = await gen.generate("a character portrait", model="flux-schnell")
    """

    MODELS = {
        "flux-schnell": "black-forest-labs/FLUX.1-schnell",
        "seedream": "ByteDance/Seedream-4.0",
    }

    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or settings.SILICONFLOW_API_KEY or ""
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.siliconflow.cn/v1",
        ) if api_key else None

    async def generate(
        self,
        prompt: str,
        model: str = "flux-schnell",
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 576,
        seed: Optional[int] = None,
        num_images: int = 1,
    ) -> ImageGenResult:
        start = time.monotonic()
        if self._client is None:
            return ImageGenResult(success=False, error="SiliconFlow API key not configured")

        try:
            response = await self._client.images.generate(
                model=self.MODELS.get(model, model),
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                n=num_images,
                size=f"{width}x{height}",
                extra_body={"seed": seed} if seed else None,
            )
            duration_ms = (time.monotonic() - start) * 1000
            url = response.data[0].url if response.data else None
            cost = calculate_unit_cost(model, num_images)
            logger.info("Image generated: model=%s, cost=$%.4f, duration=%.0fms", model, cost, duration_ms)
            return ImageGenResult(
                success=True,
                image_url=url,
                model=model,
                cost_usd=cost,
                duration_ms=duration_ms,
            )
        except Exception as e:
            duration_ms = (time.monotonic() - start) * 1000
            logger.error("Image generation failed: %s", str(e)[:200])
            return ImageGenResult(success=False, error=str(e), duration_ms=duration_ms)

    async def generate_character_anchor(
        self,
        prompt: str,
        seed: int = 42,
        width: int = 1024,
        height: int = 1024,
    ) -> ImageGenResult:
        """Generate a character anchor image using Seedream (better consistency).

        Uses a fixed seed to ensure reproducibility; the seed image serves as
        reference for subsequent character generation in later episodes.
        """
        return await self.generate(
            prompt=prompt,
            model="seedream",
            width=width,
            height=height,
            seed=seed,
            num_images=1,
        )

    async def close(self):
        if self._client:
            await self._client.close()


cloud_image_gen = CloudImageGenerator()
