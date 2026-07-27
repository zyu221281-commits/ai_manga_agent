"""Long-term memory — cross-episode context via Redis + in-memory L1 cache.

Stores character arcs, plot summaries, key events, and foreshadowing
across all episodes so later episodes maintain narrative continuity.

设计：
- L1 缓存：进程内存 dict（热数据，零延迟）
- L2 持久层：Redis Hash + Sorted Set（跨重启不丢）
- Redis 不可用时降级为纯内存模式，管线继续运行
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from app.services.cache import cache

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    key: str
    content: str
    episode_num: int
    category: str = "general"  # character / plot / event / style / foreshadowing
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class LongTermMemory:
    """Cross-episode memory for narrative continuity.

    Accumulates character states, plot developments, key events, and
    foreshadowing as the series progresses so the Writer
    agent can reference past events without repeating context.

    双层存储：
    - L1 内存 dict（本进程热缓存，重启丢失）
    - L2 Redis Hash + Sorted Set（跨重启持久化）

    Redis 不可用时降级为纯内存模式。
    """

    MAX_ENTRIES_PER_CATEGORY = 100

    # Redis key 设计：
    # Hash: mem:ep:{episode_num}:{category} → {key: json(entry)}
    # ZSet: mem:idx:{category} → {episode_num: score}（按集号索引，便于 ZREVRANGE 取最近）
    # Foreshadowing 单独用 Hash: mem:foreshadowing → {key: json(entry)}
    _REDIS_ENTRY_PREFIX = "mem:ep:"
    _REDIS_IDX_PREFIX = "mem:idx:"
    _REDIS_FORESHADOWING_KEY = "mem:foreshadowing"

    def __init__(self):
        self._store: dict[str, list[MemoryEntry]] = {
            "character": [],
            "plot": [],
            "event": [],
            "style": [],
            "foreshadowing": [],
        }

    # ================================================================
    # Redis helpers（fail-soft：失败时静默降级到内存模式）
    # ================================================================

    @staticmethod
    def _entry_to_dict(entry: MemoryEntry) -> dict:
        return {
            "key": entry.key,
            "content": entry.content,
            "episode_num": entry.episode_num,
            "category": entry.category,
            "metadata": entry.metadata,
            "created_at": entry.created_at.isoformat(),
        }

    @staticmethod
    def _dict_to_entry(d: dict) -> MemoryEntry:
        created_at_str = d.get("created_at", "")
        try:
            created_at = datetime.fromisoformat(created_at_str) if created_at_str else datetime.now(timezone.utc)
        except (ValueError, TypeError):
            created_at = datetime.now(timezone.utc)
        return MemoryEntry(
            key=d.get("key", ""),
            content=d.get("content", ""),
            episode_num=d.get("episode_num", 0),
            category=d.get("category", "general"),
            metadata=d.get("metadata", {}),
            created_at=created_at,
        )

    async def _redis_add(self, entry: MemoryEntry) -> None:
        """写入 Redis（fire-and-forget，失败静默）。"""
        try:
            if entry.category == "foreshadowing":
                # 伏笔单独存一个 Hash，便于 get_unresolved_foreshadowing 快速查询
                await cache.set(
                    f"{self._REDIS_FORESHADOWING_KEY}:{entry.key}",
                    json.dumps(self._entry_to_dict(entry)),
                    ttl=0,  # 永久（不主动过期）
                )
            else:
                # 普通记忆：Hash by episode+category，ZSet by category 索引
                hash_key = f"{self._REDIS_ENTRY_PREFIX}{entry.episode_num}:{entry.category}"
                # 用 key 字段做 hash field，value 是 JSON
                # 由于 cache.set 是简单 SET，这里用 key:field 的复合 key 模拟 Hash
                await cache.set(
                    f"{hash_key}:{entry.key}",
                    json.dumps(self._entry_to_dict(entry)),
                    ttl=0,
                )
                # ZSet 索引：score=episode_num，member=episode_num:category:key
                # cache 没有封装 zadd，这里用 SET 代替（key 包含 episode_num，可扫描）
                await cache.set(
                    f"{self._REDIS_IDX_PREFIX}{entry.category}:{entry.episode_num}:{entry.key}",
                    json.dumps({"episode_num": entry.episode_num, "category": entry.category, "key": entry.key}),
                    ttl=0,
                )
        except Exception as e:
            logger.warning("Redis write failed for memory entry (ep=%d, cat=%s): %s", entry.episode_num, entry.category, e)

    async def _redis_load_all(self) -> dict[str, list[MemoryEntry]]:
        """从 Redis 加载全部记忆到内存（启动时或缓存未命中时调用）。

        采用 SCAN 模式扫描所有 mem:* 前缀的 key。
        """
        result: dict[str, list[MemoryEntry]] = {
            "character": [], "plot": [], "event": [], "style": [], "foreshadowing": [],
        }
        try:
            r = await cache._get_redis()
            # 扫描普通记忆
            for category in ["character", "plot", "event", "style"]:
                async for key in r.scan_iter(match=f"{self._REDIS_ENTRY_PREFIX}*:{category}:*"):
                    raw = await r.get(key)
                    if not raw:
                        continue
                    try:
                        entry = self._dict_to_entry(json.loads(raw))
                        result[category].append(entry)
                    except (json.JSONDecodeError, TypeError):
                        continue
                # 按集号排序
                result[category].sort(key=lambda e: e.episode_num)
            # 扫描伏笔
            async for key in r.scan_iter(match=f"{self._REDIS_FORESHADOWING_KEY}:*"):
                raw = await r.get(key)
                if not raw:
                    continue
                try:
                    entry = self._dict_to_entry(json.loads(raw))
                    result["foreshadowing"].append(entry)
                except (json.JSONDecodeError, TypeError):
                    continue
        except Exception as e:
            logger.warning("Redis load failed, falling back to empty memory: %s", e)
        return result

    async def _redis_resolve_foreshadowing(self, key: str, episode_num: int) -> None:
        """在 Redis 中标记伏笔已解决。"""
        try:
            redis_key = f"{self._REDIS_FORESHADOWING_KEY}:{key}"
            raw = await cache.get(redis_key)
            if not raw:
                return
            d = json.loads(raw)
            d.setdefault("metadata", {})
            d["metadata"]["resolved"] = True
            d["metadata"]["resolved_in"] = episode_num
            await cache.set(redis_key, json.dumps(d), ttl=0)
        except Exception as e:
            logger.warning("Redis foreshadowing resolve failed (key=%s): %s", key, e)

    # ================================================================
    # Public API
    # ================================================================

    async def add(
        self,
        key: str,
        content: str,
        episode_num: int,
        category: str = "general",
        metadata: Optional[dict] = None,
    ) -> None:
        """Store a memory entry (双写：内存 + Redis)。"""
        if category not in self._store:
            category = "general"
            if category not in self._store:
                self._store[category] = []

        entry = MemoryEntry(
            key=key,
            content=content[:4096],
            episode_num=episode_num,
            category=category,
            metadata=metadata or {},
        )
        # L1 内存写入
        self._store[category].append(entry)
        if len(self._store[category]) > self.MAX_ENTRIES_PER_CATEGORY:
            self._store[category] = self._store[category][-self.MAX_ENTRIES_PER_CATEGORY:]

        # L2 Redis 写入（fire-and-forget）
        await self._redis_add(entry)

        logger.debug("Memory added: %s/%s (ep %d)", category, key, episode_num)

    async def get_context(
        self,
        episode_num: int,
        max_entries: int = 20,
    ) -> str:
        """Build a context summary for the Writer agent."""
        all_entries = []
        for category, entries in self._store.items():
            if category == "foreshadowing":
                continue
            for e in entries:
                if e.episode_num < episode_num:
                    all_entries.append(e)

        all_entries.sort(key=lambda e: e.episode_num, reverse=True)
        recent = all_entries[:max_entries]

        if not recent:
            return "No prior context (this is the first episode)."

        lines = [f"--- Prior Episode Context (up to episode {episode_num - 1}) ---"]
        for e in sorted(recent, key=lambda e: e.episode_num):
            lines.append(f"[Ep {e.episode_num}] [{e.category}] {e.key}: {e.content[:200]}")

        return "\n".join(lines)

    async def get_character_state(self, character_name: str) -> Optional[dict]:
        """Get the current known state of a character."""
        for entry in self._store.get("character", []):
            if entry.key == character_name:
                return {"name": character_name, "last_known": entry.content, "episode": entry.episode_num}
        return None

    async def get_recent_events(self, n: int = 5) -> list[MemoryEntry]:
        """Get the most recent plot events."""
        events = sorted(self._store.get("event", []), key=lambda e: e.episode_num, reverse=True)
        return events[:n]

    # ================================================================
    # Foreshadowing / Chekhov's Gun tracking
    # ================================================================

    async def add_foreshadowing(
        self,
        key: str,
        description: str,
        episode_num: int,
        expected_resolve_episode: Optional[int] = None,
    ) -> None:
        """Record a foreshadowing / Chekhov's gun planted in an episode."""
        metadata = {
            "planted_in": episode_num,
            "resolved_in": None,
            "resolved": False,
        }
        if expected_resolve_episode is not None:
            metadata["expected_resolve_episode"] = expected_resolve_episode
        await self.add(
            key=key,
            content=description,
            episode_num=episode_num,
            category="foreshadowing",
            metadata=metadata,
        )
        logger.info(
            "Foreshadowing '%s' planted in episode %d (resolve by ~%s)",
            key, episode_num, expected_resolve_episode or "unknown",
        )

    async def get_unresolved_foreshadowing(self) -> list[MemoryEntry]:
        """Return all unresolved foreshadowing items."""
        unresolved = []
        for entry in self._store.get("foreshadowing", []):
            if not entry.metadata.get("resolved", False):
                unresolved.append(entry)
        return unresolved

    async def resolve_foreshadowing(self, key: str, episode_num: int) -> bool:
        """Mark a foreshadowing as resolved (双写：内存 + Redis)。"""
        for entry in self._store.get("foreshadowing", []):
            if entry.key == key and not entry.metadata.get("resolved", False):
                entry.metadata["resolved"] = True
                entry.metadata["resolved_in"] = episode_num
                # 同步到 Redis
                await self._redis_resolve_foreshadowing(key, episode_num)
                logger.info("Foreshadowing '%s' resolved in episode %d", key, episode_num)
                return True
        return False

    async def get_foreshadowing_context(self, current_episode: int) -> str:
        """Build foreshadowing context for the Writer."""
        unresolved = await self.get_unresolved_foreshadowing()
        if not unresolved:
            return ""

        lines = ["--- Unresolved Foreshadowing (must be addressed) ---"]
        for entry in unresolved:
            expected = entry.metadata.get("expected_resolve_episode")
            urgency = ""
            if expected is not None and expected <= current_episode + 2:
                urgency = f" [URGENT: due by episode ~{expected}"
                if current_episode >= expected:
                    urgency += " - OVERDUE"
                urgency += "]"
            lines.append(
                f"[Planted Ep{entry.episode_num}] {entry.key}: {entry.content[:200]}{urgency}"
            )
        return "\n".join(lines)

    async def get_foreshadowing_stats(self) -> dict[str, Any]:
        """Get statistics about foreshadowing tracking."""
        all_items = self._store.get("foreshadowing", [])
        total = len(all_items)
        resolved = sum(1 for e in all_items if e.metadata.get("resolved", False))
        unresolved = total - resolved
        overdue = sum(
            1 for e in all_items
            if not e.metadata.get("resolved", False)
            and e.metadata.get("expected_resolve_episode") is not None
        )
        return {
            "total": total,
            "resolved": resolved,
            "unresolved": unresolved,
            "overdue": overdue,
            "resolution_rate": resolved / total if total > 0 else 1.0,
        }

    async def clear(self):
        """Reset all memory (for testing)."""
        for category in self._store:
            self._store[category].clear()

    async def load_from_redis(self) -> None:
        """从 Redis 加载全部记忆到内存 L1 缓存。

        在 FastAPI lifespan 或 Celery worker 启动时调用，确保跨重启不丢上下文。
        """
        loaded = await self._redis_load_all()
        total = 0
        for category, entries in loaded.items():
            if entries:
                # 合并到内存（去重：同 key 同 episode_num 的只保留一个）
                existing_keys = {(e.key, e.episode_num) for e in self._store[category]}
                for e in entries:
                    if (e.key, e.episode_num) not in existing_keys:
                        self._store[category].append(e)
                        existing_keys.add((e.key, e.episode_num))
                # 按集号排序 + 截断
                self._store[category].sort(key=lambda e: e.episode_num)
                if len(self._store[category]) > self.MAX_ENTRIES_PER_CATEGORY:
                    self._store[category] = self._store[category][-self.MAX_ENTRIES_PER_CATEGORY:]
                total += len(entries)
        if total > 0:
            logger.info("Loaded %d memory entries from Redis (L2 → L1)", total)


# Module-level singleton
long_term_memory = LongTermMemory()


async def extract_and_store_episode_memory(
    episode_num: int,
    script_data: dict[str, Any],
    episode_plan: Optional[dict[str, Any]] = None,
) -> None:
    """Extract key events, character changes, and foreshadowing from a
    completed episode script and store them in long-term memory.
    """
    scenes = script_data.get("scenes", []) if isinstance(script_data, dict) else []
    if not scenes:
        return

    characters_seen: set[str] = set()
    for scene in scenes:
        for d in scene.get("dialogue", []):
            char_name = d.get("character", "")
            if char_name and char_name not in characters_seen:
                characters_seen.add(char_name)
                expression = d.get("expression", "")
                await long_term_memory.add(
                    key=char_name,
                    content=f"Appears with expression: {expression}" if expression else "Appears in this episode",
                    episode_num=episode_num,
                    category="character",
                )

    for i, scene in enumerate(scenes):
        emotion = scene.get("emotion", "")
        narration = scene.get("narration", "")
        if narration or emotion:
            key = f"ep{episode_num}_scene_{i + 1}"
            content = f"[{emotion}] {narration}" if narration and emotion else (narration or emotion)
            await long_term_memory.add(
                key=key,
                content=content[:300],
                episode_num=episode_num,
                category="event",
            )

    if episode_plan:
        plot_summary = episode_plan.get("plot_summary", "")
        key_conflict = episode_plan.get("key_conflict", "")
        if plot_summary or key_conflict:
            await long_term_memory.add(
                key=f"ep_{episode_num}_plot",
                content=f"Summary: {plot_summary}. Conflict: {key_conflict}",
                episode_num=episode_num,
                category="plot",
            )

        hook = episode_plan.get("hook", "")
        foreshadow_keywords = ["秘密", "真相", "发现", "神秘", "伏笔", "暗示", "诡异",
                               "身份", "隐瞒", "内幕", "幕后", "真凶", "谜团"]
        combined_text = f"{hook} {key_conflict} {plot_summary}"

        if any(kw in combined_text for kw in foreshadow_keywords):
            description = f"Hook: {hook}" if hook else f"Conflict hints: {key_conflict}"
            await long_term_memory.add_foreshadowing(
                key=f"hook_ep{episode_num}",
                description=description,
                episode_num=episode_num,
                expected_resolve_episode=episode_num + 3,
            )

    logger.info("Episode %d memory stored (%d characters, %d events)",
                episode_num, len(characters_seen), len(scenes))


async def build_writer_context(episode_num: int) -> str:
    """Build a complete context string for the Writer agent."""
    prior_context = await long_term_memory.get_context(episode_num, max_entries=20)
    foreshadowing_context = await long_term_memory.get_foreshadowing_context(episode_num)

    parts = [prior_context]
    if foreshadowing_context:
        parts.append(foreshadowing_context)

    return "\n\n".join(parts)
