"""AI-漫剧 Creative Layer 测试 Runner

仅运行 3 个 Agent（无视频生成）：
  1. CreativeDirector：3 次 LLM 调用
     - 生成 3-5 个 distinct creative concepts
     - 评估选中最佳
     - 输出 creative_guidance
  2. Planner：1 次 LLM 调用（注入 creative_guidance，生成 60 集大纲）
  3. StoryCritic：双模型投票（DeepSeek + Qwen，约 2 次调用，5 维度评分）

跳过：Writer / AssetManager / Composer / Critic / SafetyCheck
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

# Windows 控制台默认 GBK，强制 stdout/stderr 走 UTF-8，避免 UnicodeEncodeError
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 切换到项目根目录，保证 .env / 模块路径正确
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from app.agents.creative_director import CreativeDirectorAgent
from app.agents.planner import PlannerAgent
from app.agents.story_critic import StoryCriticAgent
from app.services.file_lineage_tracker import FileLineageTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("creative_test")

OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)

TRACE_ID = "creative_layer_test"
SERIES_ID = "s_creative_001"
EP_ID = "ep_creative_001"

# ================================================================
# Creative Brief — 故意设计成可以衍生多种 distinct creative angles
# 题材：都市重生 + 人生模拟器系统
# 设计思路：让 CreativeDirector 可以从复仇/救赎/商业帝国/悬疑解谜/爽文等
# 多个角度切入，充分体现其多概念生成能力。
# ================================================================
CREATIVE_BRIEF = {
    "theme": "都市重生·人生模拟器",
    "genre": "都市奇幻·系统流",
    "tone": "可塑性高（待 CreativeDirector 决策）",
    "total_episodes": 30,
    "target_audience": "20-40 岁都市人群，对逆袭/遗憾救赎/职场商战感兴趣",
    "style": "cinematic_anime",
    "summary": (
        "35 岁互联网大厂程序员陈砚，连续加班 72 小时后猝死在工位。"
        "灵魂沉入黑暗之际，意外绑定『人生模拟器 2.0』系统，"
        "重生回到 2010 年大一入学第一天。"
        "系统声称：每次重大选择都会产生平行人生分支，他必须在有限次『重置』内，"
        "找到通往『真结局』的唯一路径。"
        "然而系统的真面目、上辈子那些『偶然』事件的真相、以及猝死背后的阴谋，"
        "随着一次次模拟逐渐浮出水面……"
    ),
    "core_premise": [
        "重生回 2010 年大学时代",
        "绑定『人生模拟器 2.0』，可重置次数有限",
        "每次重大选择产生平行分支",
        "需要找到通往『真结局』的唯一路径",
        "猝死背后有阴谋，逐渐揭示",
    ],
    "characters": [
        {
            "name": "陈砚",
            "role": "男主",
            "traits": ["理性", "隐忍", "前世社畜的疲惫与不甘"],
            "background": "前世 35 岁猝死程序员，重生后绑定系统",
        },
        {
            "name": "苏念",
            "role": "女主候选之一",
            "traits": ["温柔", "坚韧", "有秘密"],
            "background": "前世大学时代的初恋，因陈砚的懦弱而错过",
        },
        {
            "name": "陆景行",
            "role": "宿敌/盟友（双面）",
            "traits": ["野心家", "极度聪明", "亦正亦邪"],
            "background": "前世最终成为商业巨头，疑似与陈砚猝死有关",
        },
        {
            "name": "系统/Null",
            "role": "谜之存在",
            "traits": ["冷静", "AI 人格", "真实身份成谜"],
            "background": "『人生模拟器 2.0』，似乎在引导陈砚走向某个特定结局",
        },
    ],
    "key_conflicts": [
        "有限次重置 vs 无数平行分支",
        "弥补遗憾 vs 寻找真相",
        "信任前世好友 vs 发现他们可能参与阴谋",
        "系统指引 vs 自主选择",
    ],
    "creative_constraints": {
        "must_have": ["重置机制", "平行分支展示", "真结局伏笔"],
        "avoid": ["纯爽文无脑打脸", "后宫收集", "无逻辑系统金手指"],
        "tone_flexibility": "允许 dark/warm/epic/quirky 任一基调",
    },
}

# 热度趋势参考
HOT_TRENDS = [
    "都市重生",
    "系统流",
    "无限流/模拟器",
    "商战逆袭",
    "悬疑反转",
    "遗憾救赎",
    "AI 觉醒",
    "平行宇宙",
    "时间循环",
    "校园怀旧",
]


def save_json(name: str, data) -> str:
    p = OUTPUT / f"{name}.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return str(p)


async def main():
    total_start = time.time()
    total_cost = 0.0
    tracer = FileLineageTracker(trace_id=TRACE_ID)

    print("=" * 70)
    print("AI-Manga Drama: Creative Layer Test")
    print("(CreativeDirector → Planner → StoryCritic, NO video generation)")
    print("=" * 70)
    print(f"Theme: {CREATIVE_BRIEF['theme']}")
    print(f"Genre: {CREATIVE_BRIEF['genre']}")
    print(f"Target Episodes: {CREATIVE_BRIEF['total_episodes']}")
    print(f"Trace ID: {TRACE_ID}")
    print()

    # ================================================================
    # [1/3] CreativeDirector
    # ================================================================
    print("[1/3] CreativeDirector: generating multi-concepts + selecting + guidance...")
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
        print("  FAILED:", r.error)
        tracer.flush()
        return
    total_cost += r.cost_usd
    cd_data = r.data

    save_json("creative_layer_01_concepts", cd_data)
    print(f"  ✦ concepts generated: {cd_data.get('total_concepts', 0)}")
    selected_name = cd_data.get("selected", {}).get("concept_name", "")
    for i, c in enumerate(cd_data.get("concepts", [])):
        tone = c.get("tone", {})
        vp = c.get("viral_potential", 0)
        marker = " ★ SELECTED" if c.get("concept_name") == selected_name else ""
        print(f"    [{i}] {c.get('concept_name', '?'):<15} "
              f"tone={tone.get('primary', '?')}/{tone.get('secondary', '?')}  "
              f"viral={vp:.2f}{marker}")
        print(f"        hook: {c.get('target_hook', '')[:80]}")
    print(f"  ✦ Director note: {cd_data.get('director_note', '')[:120]}")
    guidance = cd_data.get("creative_guidance", {})
    print(f"  ✦ Guidance: tone={guidance.get('tone','?')}  "
          f"twist_freq={guidance.get('twist_frequency','?')}  "
          f"pacing={guidance.get('pacing','?')}")
    print(f"  cost=${r.cost_usd:.4f} | {r.duration_ms:.0f}ms")
    print()

    # ================================================================
    # [2/3] Planner (inject creative_guidance)
    # ================================================================
    print("[2/3] Planner: generating 60-episode outline (with creative_guidance)...")
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
        print("  FAILED:", r.error)
        tracer.flush()
        return
    total_cost += r.cost_usd
    sp = r.data
    save_json("creative_layer_02_series_plan", sp)

    eps = sp.get("episodes", [])
    print(f"  ✦ episodes generated: {len(eps)}/{sp.get('total_episodes', 0)}")
    if eps:
        e1 = eps[0]
        print(f"  ✦ Ep1: {e1.get('title','?')}  conflict={e1.get('key_conflict','')[:50]}")
        print(f"        hook: {e1.get('hook','')[:80]}")
        # 显示阶段分布
        phase1 = [e for e in eps if 1 <= e.get("episode_num", 0) <= 20]
        phase2 = [e for e in eps if 21 <= e.get("episode_num", 0) <= 40]
        phase3 = [e for e in eps if 41 <= e.get("episode_num", 0) <= 60]
        print(f"  ✦ Phase distribution: p1={len(phase1)} p2={len(phase2)} p3={len(phase3)}")
    print(f"  cost=${r.cost_usd:.4f} | {r.duration_ms:.0f}ms")
    print()

    # ================================================================
    # [3/3] StoryCritic
    # ================================================================
    print("[3/3] StoryCritic: 5-dimension quality evaluation...")
    critic = StoryCriticAgent(
        episode_id=EP_ID, series_id=SERIES_ID,
        trace_id=TRACE_ID, tracer=tracer,
    )
    r = await critic._run_with_tracking(
        series_plan=sp,
        hot_topics=HOT_TRENDS,
    )
    if not r.success:
        print("  FAILED:", r.error)
        tracer.flush()
        return
    total_cost += r.cost_usd
    ev = r.data
    save_json("creative_layer_03_critic_eval", ev)

    avg_score = ev.get("outline_score", 0)
    decision = ev.get("decision", "?")
    rewrite_eps = ev.get("rewrite_episodes", [])
    models_used = ev.get("models_used", 0)
    print(f"  ✦ outline_score: {avg_score:.2f}  decision: {decision}")
    print(f"  ✦ models voted: {models_used}  weak_episodes: {len(rewrite_eps)}")
    if rewrite_eps[:10]:
        print(f"    sample weak ep: {rewrite_eps[:10]}")
    print(f"  ✦ suggestion: {ev.get('suggestion', '')[:150]}")
    print(f"  cost=${r.cost_usd:.4f} | {r.duration_ms:.0f}ms")
    print()

    # ================================================================
    # Summary
    # ================================================================
    tracer.flush()
    total_dt = time.time() - total_start

    # 读取 trace 摘要
    trace_summary = FileLineageTracker.summary(TRACE_ID)

    print("=" * 70)
    print("CREATIVE LAYER TEST DONE")
    print("=" * 70)
    print(f"  Total duration: {total_dt:.0f}s")
    print(f"  Total cost (agent-reported): ${total_cost:.4f}")
    print(f"  Trace cost (sum from jsonl): ${trace_summary.get('total_cost_usd', 0):.4f}")
    print(f"  Trace duration: {trace_summary.get('total_duration_s', 0):.1f}s")
    print(f"  Agents executed: {trace_summary.get('agent_count', 0)}")
    print(f"  Errors: {len(trace_summary.get('errors', []))}")
    print()
    print("  Pipeline:")
    print(f"    CreativeDirector → {cd_data.get('total_concepts', 0)} concepts → "
          f"selected: {cd_data.get('selected', {}).get('concept_name', '?')}")
    print(f"    Planner → {len(eps)} episodes")
    print(f"    StoryCritic → score={avg_score:.2f} decision={decision}")
    print()
    print("  Output artifacts:")
    for f in sorted(OUTPUT.glob("creative_layer_*.json")):
        print(f"    {f}")
    print(f"  Trace file: {tracer.filepath}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
