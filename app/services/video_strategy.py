"""视频生成策略分级（ 参数化）

根据场景重要性分级选择视频生成方式，平衡成本与表现力：

- 关键场景 (KEY_SCENE)：火山方舟图生视频 ($0.30/s)
  片头/片尾、高潮反转、角色首次登场、重要情感戏
- 普通场景 (NORMAL_SCENE)：FFmpeg 静态图 + 运镜 ($0)
  Ken Burns 缩放、平移、推拉、淡入淡出转场

 关键场景比例通过 VIDEO_KEY_SCENE_RATIO 配置，支持 A/B 测试验证。
"""

from __future__ import annotations

from typing import Any

from app.core.config import settings

# 场景类型常量
KEY_SCENE = "KEY_SCENE"
NORMAL_SCENE = "NORMAL_SCENE"

# 关键场景特征词
KEY_SCENE_INDICATORS = [
    "片头", "片尾", "开场", "结尾",
    "高潮", "反转", "真相", "揭秘",
    "登场", "出场", "变身", "觉醒",
    "感情戏", "告白", "离别", "重逢",
    "战斗", "对决", "决战",
]

# 普通场景特征词
NORMAL_SCENE_INDICATORS = [
    "对话", "日常", "过渡", "回忆",
    "心理", "独白", "思考", "计划",
]


def classify_scene(scene_data: dict[str, Any], story_context: dict[str, Any] = None) -> str:
    """判断场景类型：KEY_SCENE 或 NORMAL_SCENE。

    决策逻辑：
    1. 如果 scene_data 中显式标记了 scene_type，直接使用
    2. 否则根据场景描述中的关键词判断
    3. 兜底为 NORMAL_SCENE

    Args:
        scene_data: 场景数据（emotion, narration, dialogue 等）
        story_context: 全局上下文（episode_title 等）

    Returns:
        "KEY_SCENE" 或 "NORMAL_SCENE"
    """
    # 1. 显式标记优先
    if "scene_type" in scene_data:
        st = scene_data["scene_type"]
        if st in (KEY_SCENE, NORMAL_SCENE):
            return st

    # 2. 关键词判断
    scene_text = _build_scene_text(scene_data)

    for kw in KEY_SCENE_INDICATORS:
        if kw in scene_text:
            return KEY_SCENE

    # 3. 兜底
    return NORMAL_SCENE


def get_key_scene_ratio() -> float:
    """获取关键场景比例配置。"""
    return settings.VIDEO_KEY_SCENE_RATIO


def get_scene_cost(scene_type: str, duration_s: float = 1.0) -> float:
    """计算单场景成本。"""
    if scene_type == KEY_SCENE:
        return duration_s * 0.30
    return 0.0


def is_key_scene(scene_data: dict[str, Any]) -> bool:
    """便捷方法：判断是否关键场景。"""
    return classify_scene(scene_data) == KEY_SCENE


def _build_scene_text(scene_data: dict) -> str:
    """构建场景文本用于关键词匹配。"""
    parts = []
    parts.append(scene_data.get("emotion", ""))
    parts.append(scene_data.get("narration", ""))
    for dialogue in scene_data.get("dialogue", []):
        if isinstance(dialogue, dict):
            parts.append(dialogue.get("line", ""))
            parts.append(dialogue.get("expression", ""))
            parts.append(dialogue.get("character", ""))
    return " ".join(parts)
