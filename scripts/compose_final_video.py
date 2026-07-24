"""视频合成测试 — 基于 checkpoint 恢复，仅重新生成音频 + 合成最终视频。

测试目标：
  1. 从 image.json / video.json checkpoint 恢复图像和视频（零 API 调用）
  2. 从 script.json checkpoint 恢复剧本和分镜
  3. 重新生成音频（Seed Audio 1.0），保存到 tts.json（支持断点恢复）
  4. 调用 episode_compositor 合成最终视频
  5. 验证 audio_timeline 编排能力是否正常工作

前置条件：
  - output/checkpoints/s_xianxia_001_ep_1/script.json 存在
  - output/checkpoints/s_xianxia_001_ep_1/image.json 存在
  - output/checkpoints/s_xianxia_001_ep_1/video.json 存在
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Windows 控制台 UTF-8
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import app.core  # noqa: F401  (触发 Windows EventLoop 策略)

from app.core.config import settings
from app.agents.composer import ComposerAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("compose_final")

CHECKPOINT_DIR = ROOT / "output" / "checkpoints" / "s_xianxia_001_ep_1"


def load_checkpoint_stage(stage: str) -> dict:
    """加载某个 stage 的 checkpoint。"""
    p = CHECKPOINT_DIR / f"{stage}.json"
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_script_pkg() -> dict:
    """从 script.json 加载完整 script 包（script + storyboard + image_prompts + asset_library）。"""
    data = load_checkpoint_stage("script")
    if not data:
        raise FileNotFoundError(f"script.json not found in {CHECKPOINT_DIR}")
    # checkpoint 格式：{"0": {script, storyboard, image_prompts, ...}}
    first_key = next(iter(data.keys()))
    return data[first_key]


async def main():
    print("=" * 70)
    print("视频合成测试（基于 checkpoint 恢复）")
    print("=" * 70)

    # 1. 加载 script 包
    pkg = load_script_pkg()
    script = pkg["script"]
    storyboard = pkg.get("storyboard", [])
    image_prompts = pkg.get("image_prompts", [])
    asset_library = pkg.get("asset_library") or {
        "characters": [
            {"name": "叶尘", "voice_traits": {"gender": "male"}},
            {"name": "苏挽晴", "voice_traits": {"gender": "female"}},
        ]
    }

    scenes = script.get("scenes", [])
    print(f"  Script loaded from checkpoint:")
    print(f"    scenes: {len(scenes)}")
    print(f"    storyboard shots: {len(storyboard)}")
    print(f"    image_prompts: {len(image_prompts)}")
    print(f"    characters: {[c.get('name') for c in asset_library.get('characters', [])]}")

    # 2. 检查 audio_timeline 字段
    timeline_shots = [s for s in storyboard if s.get("audio_timeline")]
    print(f"    shots with audio_timeline: {len(timeline_shots)}/{len(storyboard)}")
    if timeline_shots:
        sample = timeline_shots[0]
        print(f"    sample timeline (shot {sample.get('shot_id')}):")
        for entry in sample.get("audio_timeline", [])[:3]:
            print(f"      {entry.get('type')} {entry.get('start_s')}→{entry.get('end_s')}")

    # 3. 检查 audio_scene 字段（Seed Audio 1.0 新增）
    scene_shots = [s for s in storyboard if s.get("audio_scene")]
    print(f"    shots with audio_scene: {len(scene_shots)}/{len(storyboard)}")

    # 4. 检查 image/video checkpoint
    image_ckpt = load_checkpoint_stage("image")
    video_ckpt = load_checkpoint_stage("video")
    print(f"\n  Checkpoint status:")
    print(f"    image checkpoint: {len(image_ckpt)} shots")
    print(f"    video checkpoint: {len(video_ckpt)} shots")

    # 5. 运行 ComposerAgent（自动从 image/video checkpoint 恢复，重新生成音频）
    print(f"\n  Running ComposerAgent...")
    print(f"    AUDIO_PROVIDER: {settings.AUDIO_PROVIDER}")
    print(f"    ARK_AUDIO_MODEL: {settings.ARK_AUDIO_MODEL}")

    composer = ComposerAgent(
        episode_id="s_xianxia_001_ep_1",
        series_id="s_xianxia_001",
        trace_id="compose_final_test",
    )

    result = await composer._run_with_tracking(
        script=script,
        storyboard=storyboard,
        image_prompts=image_prompts,
        asset_library=asset_library,
    )

    if not result.success:
        print(f"\n  FAILED: {result.error}")
        return

    ea = result.data
    meta = result.metadata
    final_path = ea.get("final_video_path", "")
    final_dur = ea.get("final_video_duration_s", 0)
    final_size = ea.get("final_video_size_mb", 0)

    print(f"\n  {'=' * 50}")
    print(f"  Composition Result:")
    print(f"    images_generated: {meta.get('images_generated', 0)}")
    print(f"    videos_generated: {meta.get('videos_generated', 0)}")
    print(f"    audio_segments: {meta.get('audio_segments', 0)}")
    print(f"    subtitle_entries: {meta.get('subtitle_entries', 0)}")
    print(f"    cost: ${result.cost_usd:.4f}")
    print(f"    duration: {result.duration_ms / 1000:.1f}s")

    if final_path:
        print(f"\n  FINAL VIDEO:")
        print(f"    path: {final_path}")
        print(f"    duration: {final_dur:.1f}s")
        print(f"    size: {final_size:.2f} MB")
        print(f"\n  [SUCCESS] 视频合成完成！")
    else:
        print(f"\n  [WARNING] 无最终视频输出")

    # 6. 验证 audio_timeline 是否被使用
    composition = meta.get("composition", {})
    if isinstance(composition, dict):
        print(f"\n  Composition details:")
        print(f"    {json.dumps(composition, ensure_ascii=False, indent=2)[:500]}")

    print(f"\n{'=' * 70}")
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
