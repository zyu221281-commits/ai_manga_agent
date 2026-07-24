"""AI-漫剧 完整 7-Agent 管道测试

**DEV ONLY**: 手动验证用。生产走 Celery Beat + Worker。

跑全部 Agent 链：
  1. CreativeDirector → 4 concepts + creative_guidance
  2. Planner         → 1 集大纲（强制单集）
  3. StoryCritic     → 大纲评分
  4. AssetManager    → 角色资产 + 风格模板
  5. Writer          → 剧本 + 分镜 + 提示词
  6. Composer        → 图像 + 视频 + TTS + 合成 MP4
  7. Critic          → 单集质量评估

题材：霓虹症候群（赛博朋克失忆侦探短剧，单集 60-90s）
统计：每个 Agent 的 LLM token 用量 + 总成本
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
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

from app.agents.creative_director import CreativeDirectorAgent
from app.agents.planner import PlannerAgent
from app.agents.story_critic import StoryCriticAgent
from app.agents.asset_manager import AssetManagerAgent
from app.agents.writer import WriterAgent
from app.agents.composer import ComposerAgent
from app.agents.critic import CriticAgent
from app.services.file_lineage_tracker import FileLineageTracker
from app.services.llm_client import llm_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("full_pipeline")

OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)
(OUTPUT / "images").mkdir(exist_ok=True)
(OUTPUT / "videos").mkdir(exist_ok=True)
(OUTPUT / "audio").mkdir(exist_ok=True)

TRACE_ID = "full_pipeline_test"
SERIES_ID = "s_neon_001"
EP_ID = "ep_neon_001"


# ================================================================
# 简报：霓虹症候群
# 设计目标：单集 60-90s，3 角色，5 scene，赛博朋克视觉
# ================================================================
CREATIVE_BRIEF = {
    "theme": "霓虹症候群",
    "genre": "赛博朋克·悬疑侦探",
    "tone": "dark·neo-noir",
    "total_episodes": 1,  # 单集测试
    "target_audience": "20-35 岁喜爱赛博朋克/悬疑的群体",
    "style": "cyberpunk_anime",
    "summary": (
        "2147 年新上海，记忆可以被编辑、贩卖、销毁。"
        "私家侦探 K 醒来时手里握着一把带血的小刀，身旁是一具他不认识的尸体。"
        "他的电子脑只残留一段 17 秒的录音：一个女人在喊『别相信镜子里的自己』。"
        "他必须在 24 小时内找出真相——凶手究竟是自己，还是被人植入的虚假人格？"
    ),
    "core_premise": [
        "记忆可被编辑/贩卖的近未来",
        "主角失忆，疑似凶手",
        "17 秒神秘录音是唯一线索",
        "镜子意象贯穿全剧",
        "真凶身份成谜",
    ],
    "characters": [
        {
            "name": "K",
            "role": "男主",
            "traits": ["冷峻", "理性", "自我怀疑"],
            "background": "私家侦探，电子脑被篡改",
        },
        {
            "name": "镜",
            "role": "女主/谜之存在",
            "traits": ["空灵", "神秘", "亦正亦邪"],
            "background": "录音中的女人，身份不明",
        },
        {
            "name": "尸/林博士",
            "role": "关键配角",
            "traits": ["已死", "记忆科学家"],
            "background": "受害者，记忆技术发明者",
        },
    ],
    "key_conflicts": [
        "失忆自保 vs 寻找真相",
        "信任自我 vs 怀疑自我",
        "记忆真实 vs 被植入",
    ],
    "creative_constraints": {
        "must_have": ["17 秒录音", "镜子意象", "雨夜霓虹场景"],
        "avoid": ["长篇背景说明", "超过 5 个 scene"],
        "tone_flexibility": "允许 dark/quirky 任一",
    },
    "episode_target": {
        "duration_s": 75,
        "scene_count": 5,
        "dialogue_chars": 240,
    },
}

HOT_TRENDS = [
    "赛博朋克",
    "记忆篡改",
    "失忆悬疑",
    "neo-noir",
    "电子脑",
    "镜子隐喻",
    "短剧悬疑",
]


# ================================================================
# Token 统计：包装 LLM 客户端
# ================================================================
class TokenStats:
    """聚合 LLM 调用的 token 统计。"""

    def __init__(self):
        self.calls: list[dict] = []
        self.total_input = 0
        self.total_output = 0
        self.total_cost = 0.0

    def record(self, agent: str, model: str, input_tokens: int, output_tokens: int, cost_usd: float):
        self.calls.append({
            "agent": agent,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
        })
        self.total_input += input_tokens
        self.total_output += output_tokens
        self.total_cost += cost_usd

    def by_agent(self) -> dict:
        agg: dict[str, dict] = {}
        for c in self.calls:
            a = c["agent"]
            if a not in agg:
                agg[a] = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
            agg[a]["calls"] += 1
            agg[a]["input_tokens"] += c["input_tokens"]
            agg[a]["output_tokens"] += c["output_tokens"]
            agg[a]["cost_usd"] += c["cost_usd"]
        return agg

    def summary(self) -> dict:
        return {
            "total_calls": len(self.calls),
            "total_input_tokens": self.total_input,
            "total_output_tokens": self.total_output,
            "total_tokens": self.total_input + self.total_output,
            "total_cost_usd": round(self.total_cost, 6),
            "by_agent": self.by_agent(),
        }


# 全局 token 统计器
TOKEN_STATS = TokenStats()


def install_token_counter():
    """Patch llm_client.completions 自动记录 token。"""
    original = llm_client.completions

    async def wrapped(*args, **kwargs):
        resp = await original(*args, **kwargs)
        # 从调用栈中推断 agent_name（最近一帧的 self.agent_name）
        agent_name = "unknown"
        try:
            frame = sys._getframe(1)
            while frame:
                self_var = frame.f_locals.get("self")
                if self_var and hasattr(self_var, "agent_name"):
                    agent_name = self_var.agent_name
                    break
                frame = frame.f_back
        except Exception:
            pass
        TOKEN_STATS.record(
            agent=agent_name,
            model=resp.model,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            cost_usd=resp.cost_usd,
        )
        return resp

    llm_client.completions = wrapped


def save_json(name: str, data) -> str:
    p = OUTPUT / f"{name}.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return str(p)


async def main():
    install_token_counter()
    total_start = time.time()
    total_cost = 0.0
    tracer = FileLineageTracker(trace_id=TRACE_ID)

    print("=" * 70)
    print("AI-Manga Drama: FULL 7-Agent Pipeline Test")
    print("CD → Planner → StoryCritic → AssetManager → Writer → Composer → Critic")
    print("=" * 70)
    print(f"Theme: {CREATIVE_BRIEF['theme']}")
    print(f"Genre: {CREATIVE_BRIEF['genre']}")
    print(f"Target: {CREATIVE_BRIEF['episode_target']['duration_s']}s, "
          f"{CREATIVE_BRIEF['episode_target']['scene_count']} scenes")
    print(f"Trace ID: {TRACE_ID}")
    print()

    # ================================================================
    # [1/7] CreativeDirector
    # ================================================================
    print("[1/7] CreativeDirector: multi-concepts + guidance...")
    cd = CreativeDirectorAgent(
        episode_id=EP_ID, series_id=SERIES_ID,
        trace_id=TRACE_ID, tracer=tracer,
    )
    r = await cd._run_with_tracking(
        creative_brief=CREATIVE_BRIEF,
        hot_trends=HOT_TRENDS,
        num_concepts=4,
    )
    if not r.success:
        print("  FAILED:", r.error); tracer.flush(); return
    total_cost += r.cost_usd
    cd_data = r.data
    save_json("full_01_concepts", cd_data)
    selected = cd_data.get("selected", {})
    guidance = cd_data.get("creative_guidance", {})
    print(f"  concepts: {cd_data.get('total_concepts', 0)} | "
          f"selected: {selected.get('concept_name', '?')} | "
          f"viral={selected.get('viral_potential', 0):.2f}")
    print(f"  guidance tone={guidance.get('tone', '?')} "
          f"pacing={guidance.get('pacing', '?')} | "
          f"${r.cost_usd:.4f} | {r.duration_ms:.0f}ms")
    print()

    # ================================================================
    # [2/7] Planner
    # ================================================================
    print("[2/7] Planner: generating outline (single episode)...")
    planner = PlannerAgent(
        episode_id=EP_ID, series_id=SERIES_ID,
        trace_id=TRACE_ID, tracer=tracer,
    )
    r = await planner._run_with_tracking(
        creative_brief=CREATIVE_BRIEF,
        hot_trends=HOT_TRENDS,
        creative_guidance=guidance,
    )
    if not r.success:
        print("  FAILED:", r.error); tracer.flush(); return
    total_cost += r.cost_usd
    sp = r.data
    # 强制只取第 1 集（即使 Planner 生成了多集）
    all_eps = sp.get("episodes", [])
    if len(all_eps) > 1:
        sp["episodes"] = all_eps[:1]
        sp["total_episodes"] = 1
        logger.info("Truncated to single episode (was %d)", len(all_eps))
    # 覆盖单集目标时长
    if sp["episodes"]:
        sp["episodes"][0]["estimated_duration"] = CREATIVE_BRIEF["episode_target"]["duration_s"]
    save_json("full_02_series_plan", sp)
    ep1 = sp["episodes"][0] if sp.get("episodes") else {}
    print(f"  ep1: {ep1.get('title', '?')} | duration={ep1.get('estimated_duration', 0)}s")
    print(f"  conflict: {ep1.get('key_conflict', '')[:60]}")
    print(f"  ${r.cost_usd:.4f} | {r.duration_ms:.0f}ms")
    print()

    # ================================================================
    # [3/7] StoryCritic
    # ================================================================
    print("[3/7] StoryCritic: outline evaluation...")
    critic_eval = StoryCriticAgent(
        episode_id=EP_ID, series_id=SERIES_ID,
        trace_id=TRACE_ID, tracer=tracer,
    )
    r = await critic_eval._run_with_tracking(
        series_plan=sp,
        hot_topics=HOT_TRENDS,
    )
    if not r.success:
        print("  FAILED:", r.error); tracer.flush(); return
    total_cost += r.cost_usd
    ev = r.data
    save_json("full_03_critic_eval", ev)
    print(f"  score={ev.get('outline_score', 0):.2f} "
          f"decision={ev.get('decision', '?')} | "
          f"${r.cost_usd:.4f} | {r.duration_ms:.0f}ms")
    print()

    # ================================================================
    # [4/7] AssetManager
    # ================================================================
    print("[4/7] AssetManager: character assets + style template...")
    am = AssetManagerAgent(
        episode_id=EP_ID, series_id=SERIES_ID,
        trace_id=TRACE_ID, tracer=tracer,
    )
    r = await am._run_with_tracking(
        series_plan=sp,
        characters=CREATIVE_BRIEF.get("characters"),
    )
    if not r.success:
        print("  FAILED:", r.error); tracer.flush(); return
    total_cost += r.cost_usd
    al = r.data
    save_json("full_04_asset_library", al)
    st = al.get("style_template", {})
    print(f"  chars: {len(al.get('characters', []))} | "
          f"style: {st.get('name', '?')} | "
          f"${r.cost_usd:.4f} | {r.duration_ms:.0f}ms")
    print()

    # ================================================================
    # [5/7] Writer
    # ================================================================
    print("[5/7] Writer: script + storyboard + prompts...")
    writer = WriterAgent(
        episode_id=EP_ID, series_id=SERIES_ID,
        trace_id=TRACE_ID, tracer=tracer,
    )
    r = await writer._run_with_tracking(
        episode_plan=ep1,
        character_anchors={c["name"]: c for c in al.get("characters", [])},
        style_template=st,
    )
    if not r.success:
        print("  FAILED:", r.error); tracer.flush(); return
    total_cost += r.cost_usd
    pkg = r.data
    save_json("full_05_script_package", pkg)
    scenes = pkg.get("script", {}).get("scenes", [])
    shots = pkg.get("storyboard", [])
    prompts = pkg.get("image_prompts", [])
    print(f"  scenes={len(scenes)} shots={len(shots)} prompts={len(prompts)} | "
          f"${r.cost_usd:.4f} | {r.duration_ms:.0f}ms")
    # 校验总时长
    total_dur = sum(s.get("duration_s", 0) for s in scenes)
    print(f"  total scene duration: {total_dur}s (target: {CREATIVE_BRIEF['episode_target']['duration_s']}s)")
    print()

    # ================================================================
    # [6/7] Composer
    # ================================================================
    print("[6/7] Composer: images + videos + TTS + FFmpeg composition...")
    composer = ComposerAgent(
        episode_id=EP_ID, series_id=SERIES_ID,
        trace_id=TRACE_ID, tracer=tracer,
    )
    r = await composer._run_with_tracking(
        script=pkg["script"],
        storyboard=pkg["storyboard"],
        image_prompts=pkg["image_prompts"],
        asset_library=al,
    )
    if not r.success:
        print("  FAILED:", r.error); tracer.flush(); return
    total_cost += r.cost_usd
    ea = r.data
    save_json("full_06_episode_asset", ea)
    meta = r.metadata
    final_path = ea.get("final_video_path", "")
    final_dur = ea.get("final_video_duration_s", 0)
    final_size = ea.get("final_video_size_mb", 0)
    print(f"  images={meta.get('images_generated', 0)} "
          f"videos={meta.get('videos_generated', 0)} "
          f"audio={meta.get('audio_segments', 0)} | "
          f"${r.cost_usd:.4f} | {r.duration_ms:.0f}ms")
    if final_path:
        print(f"  FINAL VIDEO: {final_path}")
        print(f"    duration={final_dur:.1f}s | size={final_size:.2f} MB")
    else:
        print("  WARNING: no final video composed")
    print()

    # ================================================================
    # [7/7] Critic
    # ================================================================
    print("[7/7] Critic: episode quality evaluation...")
    cr = CriticAgent(
        episode_id=EP_ID, series_id=SERIES_ID,
        trace_id=TRACE_ID, tracer=tracer,
    )
    r = await cr._run_with_tracking(
        episode_asset=ea,
    )
    total_cost += r.cost_usd
    ev_final = r.data if r.success else {}
    save_json("full_07_critic_evaluation", ev_final)
    score = ev_final.get("overall_score", 0)
    verdict = ev_final.get("verdict", "?")
    dims = ev_final.get("dimensions", {})
    print(f"  score={score:.2f} verdict={verdict} | "
          f"${r.cost_usd:.4f} | {r.duration_ms:.0f}ms")
    for k, v in sorted(dims.items()):
        bar = "#" * int(v * 20) + "-" * (20 - int(v * 20))
        print(f"    {k:<24}: [{bar}] {v:.2f}")
    print()

    # ================================================================
    # Summary
    # ================================================================
    tracer.flush()
    total_dt = time.time() - total_start
    token_summary = TOKEN_STATS.summary()

    print("=" * 70)
    print("FULL 7-AGENT PIPELINE DONE")
    print("=" * 70)
    print(f"  Total wall time: {total_dt:.0f}s ({total_dt/60:.1f} min)")
    print(f"  Total cost (agent-reported): ${total_cost:.4f}")
    print()
    print("  TOKEN USAGE BY AGENT:")
    print(f"  {'Agent':<22} {'Calls':>6} {'Input':>10} {'Output':>10} {'Cost$':>10}")
    print(f"  {'-'*22} {'-'*6} {'-'*10} {'-'*10} {'-'*10}")
    for agent, stats in token_summary["by_agent"].items():
        print(f"  {agent:<22} {stats['calls']:>6} {stats['input_tokens']:>10} "
              f"{stats['output_tokens']:>10} {stats['cost_usd']:>10.4f}")
    print(f"  {'-'*22} {'-'*6} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'TOTAL':<22} {token_summary['total_calls']:>6} "
          f"{token_summary['total_input_tokens']:>10} "
          f"{token_summary['total_output_tokens']:>10} "
          f"{token_summary['total_cost_usd']:>10.4f}")
    print()
    print("  PIPELINE RESULTS:")
    print(f"    [1] CreativeDirector → {cd_data.get('total_concepts', 0)} concepts → "
          f"selected: {selected.get('concept_name', '?')}")
    print(f"    [2] Planner → {sp.get('total_episodes', 0)} episode(s)")
    print(f"    [3] StoryCritic → score={ev.get('outline_score', 0):.2f} "
          f"decision={ev.get('decision', '?')}")
    print(f"    [4] AssetManager → {len(al.get('characters', []))} chars")
    print(f"    [5] Writer → {len(scenes)} scenes | {len(shots)} shots | {len(prompts)} prompts")
    print(f"    [6] Composer → {meta.get('images_generated', 0)} images / "
          f"{meta.get('videos_generated', 0)} videos / {meta.get('audio_segments', 0)} audio")
    print(f"    [7] Critic → score={score:.2f} verdict={verdict}")
    print()
    if final_path:
        print(f"  FINAL EPISODE: {final_path}")
        print(f"    duration: {final_dur:.1f}s | size: {final_size:.2f} MB")
    print()
    print("  OUTPUT ARTIFACTS:")
    for f in sorted(OUTPUT.glob("full_*.json")):
        print(f"    {f}")
    print(f"  TRACE FILE: {tracer.filepath}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
