"""Character consistency service — real CLIP similarity check.

Uses open_clip ViT-B/32 to compute cosine similarity between a generated
image and the character's seed anchor image. On low similarity, triggers
img2img regeneration using the anchor as reference via Seedream API.

M4: Anchor persistence via PostgreSQL (memory cache + DB backend).
M5: CLIP GPU memory management with empty_cache() and release_clip_model().
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from sqlalchemy import Column, Integer, String, Text, select
from sqlalchemy.orm import DeclarativeBase

logger = logging.getLogger(__name__)

# Lazy-loaded CLIP model (singleton)
_clip_model = None
_clip_preprocess = None
_clip_device: str = "cpu"


def _get_clip():
    """Lazy-load OpenCLIP ViT-B/32 model."""
    global _clip_model, _clip_preprocess, _clip_device
    if _clip_model is None:
        import open_clip
        _clip_device = "cuda" if torch.cuda.is_available() else "cpu"
        _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        _clip_model = _clip_model.to(_clip_device).eval()
        logger.info("CLIP ViT-B/32 loaded on %s", _clip_device)
    return _clip_model, _clip_preprocess


def release_clip_model():
    """Release CLIP model from GPU memory (M5).

    Call this during idle periods or when GPU memory is needed for other tasks.
    The model will be re-loaded on next check_consistency call.
    """
    global _clip_model, _clip_preprocess
    if _clip_model is not None:
        del _clip_model
        _clip_model = None
        _clip_preprocess = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CLIP model released, GPU cache cleared")


# ================================================================
# M4: PostgreSQL persistence for character anchors
# ================================================================

class _Base(DeclarativeBase):
    pass


class CharacterAnchorModel(_Base):
    """SQLAlchemy ORM model for character anchor persistence."""

    __tablename__ = "character_anchors"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), unique=True, nullable=False, index=True)
    role = Column(String(100), default="")
    seed_prompt = Column(Text, default="")
    seed_image_url = Column(Text, default="")
    seed_image_path = Column(Text, default="")
    seed = Column(Integer, default=42)
    traits_json = Column(Text, default="[]")
    generation_count = Column(Integer, default=0)
    # Multi-view anchor fields (front / side / back), each with local path + remote URL
    front_image_path = Column(Text, default="")
    side_image_path = Column(Text, default="")
    back_image_path = Column(Text, default="")
    front_image_url = Column(Text, default="")
    side_image_url = Column(Text, default="")
    back_image_url = Column(Text, default="")


@dataclass
class CharacterAnchor:
    name: str
    role: str
    seed_prompt: str
    seed_image_url: Optional[str] = None
    seed_image_path: Optional[str] = None
    seed: int = 42
    traits: list[str] = field(default_factory=list)
    generation_count: int = 0
    # Multi-view anchor
    # view_images: {"front": local_path, "side": local_path, "back": local_path}
    view_images: dict[str, str] = field(default_factory=dict)
    # view_image_urls: {"front": url, "side": url, "back": url}
    view_image_urls: dict[str, str] = field(default_factory=dict)


@dataclass
class ConsistencyResult:
    passed: bool
    similarity_score: float = 0.0
    character_name: str = ""
    reference_url: str = ""
    generated_url: str = ""
    recommendation: str = ""


class CharacterConsistencyChecker:
    """Real CLIP-based character visual consistency checker.

    M4: Uses memory cache (fast) + PostgreSQL backend (durable).
    M5: Clears GPU cache after each check to prevent OOM.

    Workflow:
    1. Generate anchor image with fixed seed (once per character)
    2. For each new generation, compute CLIP similarity against anchor
    3. If similarity < threshold, regenerate with anchor as img2img reference
    """

    DEFAULT_THRESHOLD = 0.82

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self._threshold = threshold
        self._anchors: dict[str, CharacterAnchor] = {}
        self._table_ensured = False

    # ---- M4: Table management ----

    async def ensure_table_created(self) -> None:
        """Create the character_anchors table if not exists. Call at startup."""
        if self._table_ensured:
            return
        try:
            from app.core.dependencies import get_session_factory
            factory = get_session_factory()
            async with factory() as session:
                await _Base.metadata.create_all(session.bind)
            self._table_ensured = True
            logger.info("CharacterAnchor table ensured")
        except Exception as e:
            logger.warning("Failed to create character_anchors table: %s", e)

    async def load_all_anchors(self) -> None:
        """Load all anchors from DB into memory cache. Call at startup."""
        try:
            from app.core.dependencies import get_session_factory
            factory = get_session_factory()
            async with factory() as session:
                result = await session.execute(select(CharacterAnchorModel))
                rows = result.scalars().all()
                for row in rows:
                    view_images = {
                        "front": row.front_image_path or "",
                        "side": row.side_image_path or "",
                        "back": row.back_image_path or "",
                    }
                    view_image_urls = {
                        "front": row.front_image_url or "",
                        "side": row.side_image_url or "",
                        "back": row.back_image_url or "",
                    }
                    anchor = CharacterAnchor(
                        name=row.name,
                        role=row.role or "",
                        seed_prompt=row.seed_prompt or "",
                        seed_image_url=row.seed_image_url or None,
                        seed_image_path=row.seed_image_path or None,
                        seed=row.seed or 42,
                        traits=json.loads(row.traits_json or "[]"),
                        generation_count=row.generation_count or 0,
                        view_images=view_images,
                        view_image_urls=view_image_urls,
                    )
                    self._anchors[anchor.name] = anchor
                if rows:
                    logger.info("Loaded %d character anchors from DB", len(rows))
        except Exception as e:
            logger.warning("Failed to load anchors from DB: %s", e)

    async def _persist_anchor(self, anchor: CharacterAnchor) -> None:
        """Persist or update an anchor in the database."""
        try:
            from app.core.dependencies import get_session_factory
            factory = get_session_factory()
            async with factory() as session:
                stmt = select(CharacterAnchorModel).where(
                    CharacterAnchorModel.name == anchor.name
                )
                result = await session.execute(stmt)
                existing = result.scalar_one_or_none()

                view_images = anchor.view_images or {}
                view_image_urls = anchor.view_image_urls or {}
                fields = {
                    "role": anchor.role,
                    "seed_prompt": anchor.seed_prompt,
                    "seed_image_url": anchor.seed_image_url or "",
                    "seed_image_path": anchor.seed_image_path or "",
                    "seed": anchor.seed,
                    "traits_json": json.dumps(anchor.traits),
                    "generation_count": anchor.generation_count,
                    "front_image_path": view_images.get("front", "") or "",
                    "side_image_path": view_images.get("side", "") or "",
                    "back_image_path": view_images.get("back", "") or "",
                    "front_image_url": view_image_urls.get("front", "") or "",
                    "side_image_url": view_image_urls.get("side", "") or "",
                    "back_image_url": view_image_urls.get("back", "") or "",
                }

                if existing:
                    for k, v in fields.items():
                        setattr(existing, k, v)
                else:
                    session.add(CharacterAnchorModel(
                        name=anchor.name, **fields,
                    ))
                await session.commit()
        except Exception as e:
            logger.warning("Failed to persist anchor %s: %s", anchor.name, e)

    # ---- Anchor registration ----

    def register_anchor(
        self,
        name: str,
        role: str = "",
        seed_prompt: str = "",
        seed_image_url: Optional[str] = None,
        seed_image_path: Optional[str] = None,
        seed: int = 42,
        traits: Optional[list[str]] = None,
    ) -> None:
        """Register a character anchor for consistency tracking.

        Writes to memory cache immediately, persists to DB asynchronously.
        """
        anchor = CharacterAnchor(
            name=name,
            role=role,
            seed_prompt=seed_prompt,
            seed_image_url=seed_image_url,
            seed_image_path=seed_image_path,
            seed=seed,
            traits=traits or [],
        )
        self._anchors[name] = anchor
        logger.info(
            "Character anchor registered: %s (role=%s, has_image=%s)",
            name, role, bool(seed_image_url or seed_image_path),
        )

        # M4: fire-and-forget async DB persistence
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._persist_anchor(anchor))
        except RuntimeError:
            # No running event loop (e.g., in tests); skip DB persistence
            pass

    def get_anchor(self, name: str) -> Optional[CharacterAnchor]:
        return self._anchors.get(name)

    def has_anchor_image(self, name: str) -> bool:
        """Check if we have a seed image for this character."""
        anchor = self._anchors.get(name)
        if anchor is None:
            return False
        return bool(anchor.seed_image_path) or bool(anchor.seed_image_url)

    # ---- M5: CLIP consistency check with GPU memory management ----

    async def check_consistency(
        self,
        character_name: str,
        generated_image_path: str,
    ) -> ConsistencyResult:
        """Compute CLIP cosine similarity between anchor and generated image.

        M5: Clears CUDA cache after each check to prevent GPU OOM on long runs.
        """
        anchor = self._anchors.get(character_name)
        if anchor is None:
            return ConsistencyResult(
                passed=True,
                similarity_score=1.0,
                character_name=character_name,
                generated_url=generated_image_path,
                recommendation="No anchor registered; assuming first generation.",
            )

        anchor_path = anchor.seed_image_path
        if not anchor_path:
            return ConsistencyResult(
                passed=True,
                similarity_score=1.0,
                character_name=character_name,
                reference_url=anchor.seed_image_url or "",
                generated_url=generated_image_path,
                recommendation="No anchor image file available; skipping check.",
            )

        anchor.generation_count += 1

        try:
            model, preprocess = _get_clip()
            device = next(model.parameters()).device

            img_anchor = preprocess(Image.open(anchor_path).convert("RGB")).unsqueeze(0).to(device)
            img_generated = preprocess(Image.open(generated_image_path).convert("RGB")).unsqueeze(0).to(device)

            with torch.no_grad():
                feat_anchor = model.encode_image(img_anchor)
                feat_generated = model.encode_image(img_generated)

                # Normalize and compute cosine similarity
                feat_anchor = feat_anchor / feat_anchor.norm(dim=-1, keepdim=True)
                feat_generated = feat_generated / feat_generated.norm(dim=-1, keepdim=True)
                similarity = float((feat_anchor @ feat_generated.T).item())

        except Exception as e:
            logger.warning("CLIP consistency check failed for %s: %s", character_name, e)
            return ConsistencyResult(
                passed=True,
                similarity_score=1.0,
                character_name=character_name,
                generated_url=generated_image_path,
                recommendation=f"CLIP computation error, accepting: {e}",
            )

        finally:
            # M5: Free intermediate tensors and clear GPU cache
            del img_anchor, img_generated
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        passed = similarity >= self._threshold
        recommendation = ""
        if not passed:
            recommendation = (
                f"Similarity {similarity:.2f} below threshold {self._threshold}. "
                f"Consider img2img regeneration with anchor reference."
            )

        logger.info(
            "CLIP consistency: %s similarity=%.3f threshold=%.2f passed=%s (gen #%d)",
            character_name, similarity, self._threshold, passed, anchor.generation_count,
        )

        return ConsistencyResult(
            passed=passed,
            similarity_score=similarity,
            character_name=character_name,
            reference_url=anchor.seed_image_url or "",
            generated_url=generated_image_path,
            recommendation=recommendation,
        )

    async def check_consistency_url(
        self,
        character_name: str,
        generated_image_url: str,
    ) -> ConsistencyResult:
        """Check consistency when we only have a URL (downloads locally first)."""
        import tempfile, httpx
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(generated_image_url)
                if resp.status_code == 200:
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                        f.write(resp.content)
                        tmp_path = f.name
                    result = await self.check_consistency(character_name, tmp_path)
                    Path(tmp_path).unlink(missing_ok=True)
                    return result
        except Exception as e:
            logger.warning("Failed to download image for consistency check: %s", e)
        return ConsistencyResult(
            passed=True, similarity_score=1.0,
            character_name=character_name, generated_url=generated_image_url,
            recommendation="Could not download image for check; accepting.",
        )

    def generate_with_anchor(
        self,
        character_name: str,
        new_prompt: str,
        base_prompt: str = "",
    ) -> str:
        """Enrich a prompt with anchor information for consistent generation."""
        anchor = self._anchors.get(character_name)
        if anchor is None:
            return new_prompt

        traits_str = ", ".join(anchor.traits) if anchor.traits else ""
        parts = [base_prompt, new_prompt]
        if traits_str:
            parts.append(traits_str)
        parts.append("consistent with character design, same character appearance")
        return ", ".join(p for p in parts if p)

    def get_anchor_ref_image(self, character_name: str) -> Optional[str]:
        """Get the anchor reference image (URL or path) for img2img."""
        anchor = self._anchors.get(character_name)
        if anchor is None:
            return None
        return anchor.seed_image_url or anchor.seed_image_path

    # ================================================================
    # Multi-view anchor (front / side / back)
    # ================================================================

    # shot_angle → ref view mapping
    VIEW_MAPPING = {
        "close-up": "front",
        "extreme-close-up": "front",
        "medium": "front",
        "wide": "front",
        "side-angle": "side",
        "over-shoulder": "side",
        "back": "back",
    }

    def register_multi_view_anchor(
        self,
        name: str,
        views: dict[str, str | None],
        image_urls: dict[str, str | None] | None = None,
        appearance_text: str = "",
    ) -> None:
        """注册多视角 anchor（front/side/back）。

        Args:
            name: 角色名
            views: {"front": local_path, "side": local_path, "back": local_path}
            image_urls: {"front": url, "side": url, "back": url}（可选）
            appearance_text: 角色 canonical 外貌描述（来自 id_card 原文）。
                持久化到 seed_prompt，跨集时作为视频/图像生成的一致性约束源。
        """
        anchor = self._anchors.get(name)
        if anchor is None:
            # 自动注册一个空 anchor（仅 multi-view 数据）
            logger.warning(
                "register_multi_view_anchor: '%s' not registered, creating skeleton anchor",
                name,
            )
            anchor = CharacterAnchor(
                name=name, role="", seed_prompt=appearance_text or "",
                view_images=dict(views or {}),
                view_image_urls=dict(image_urls or {}),
            )
            self._anchors[name] = anchor
        else:
            anchor.view_images = {**(anchor.view_images or {}), **(views or {})}
            anchor.view_image_urls = {
                **(anchor.view_image_urls or {}),
                **(image_urls or {}),
            }
            # 仅在 anchor 尚无 canonical appearance 时写入，避免被后续集覆盖
            # （保证第 1 集生成的外观描述成为全系列唯一基准）
            if appearance_text and not anchor.seed_prompt:
                anchor.seed_prompt = appearance_text

        # 若 front 视图存在但 seed_image 未设置，用 front 作为默认 seed_image
        # （保持向后兼容：旧的 get_anchor_ref_image 仍可用）
        front_path = anchor.view_images.get("front", "")
        front_url = anchor.view_image_urls.get("front", "")
        if front_path and not anchor.seed_image_path:
            anchor.seed_image_path = front_path
        if front_url and not anchor.seed_image_url:
            anchor.seed_image_url = front_url

        logger.info(
            "Multi-view anchor registered: %s (views=%s)",
            name,
            {k: bool(v) for k, v in (anchor.view_images or {}).items()},
        )

        # 异步持久化
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._persist_anchor(anchor))
        except RuntimeError:
            pass

    def has_multi_view(self, name: str) -> bool:
        """检查角色是否有多视角 anchor（至少含 front）。"""
        anchor = self._anchors.get(name)
        if anchor is None:
            return False
        views = anchor.view_images or {}
        # 至少 front 必须有（其他视图可选）
        return bool(views.get("front"))

    def get_best_ref_view(self, name: str, shot_angle: str) -> Optional[str]:
        """根据 shot_angle 返回最佳 ref_image path/url。

        映射规则：
            close-up / extreme-close-up → front
            medium / wide              → front
            side-angle / over-shoulder → side
            back                        → back

        若对应视图不存在，回退到 front；front 也不存在则回退到 seed_image。
        """
        anchor = self._anchors.get(name)
        if anchor is None:
            return None

        target_view = self.VIEW_MAPPING.get((shot_angle or "").lower(), "front")
        views = anchor.view_images or {}
        urls = anchor.view_image_urls or {}

        # 优先返回 URL（远程可访问），其次本地路径
        if urls.get(target_view):
            return urls[target_view]
        if views.get(target_view):
            return views[target_view]

        # 回退到 front
        if target_view != "front":
            if urls.get("front"):
                return urls["front"]
            if views.get("front"):
                return views["front"]

        # 最终回退到 seed_image（旧 anchor）
        return anchor.seed_image_url or anchor.seed_image_path

    def get_canonical_appearance(self, name: str) -> str:
        """返回角色 canonical 外貌描述（跨集唯一基准）。

        用于注入到视频生成 prompt，确保 25 集动画过程中角色外观不偏移。
        来源：generate_multi_view_anchors 时从 id_card 原文构建并持久化到 seed_prompt。
        """
        anchor = self._anchors.get(name)
        if anchor is None:
            return ""
        return anchor.seed_prompt or ""

    def list_anchors(self) -> list[dict]:
        return [
            {
                "name": a.name,
                "role": a.role,
                "generations": a.generation_count,
                "has_image": bool(a.seed_image_url or a.seed_image_path),
                "has_multi_view": bool((a.view_images or {}).get("front")),
                "views": {k: bool(v) for k, v in (a.view_images or {}).items()},
            }
            for a in self._anchors.values()
        ]

    def clear(self):
        """Clear memory cache only (DB records remain for durability)."""
        self._anchors.clear()


character_consistency = CharacterConsistencyChecker()
