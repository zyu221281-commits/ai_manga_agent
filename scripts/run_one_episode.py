"""AI-漫剧 单集完整管线 Runner V5 — 通过 LangGraph graph.ainvoke() 执行"""

import asyncio, json, os, sys, time, logging
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent)
sys.path.insert(0, str(Path.cwd()))

from app.state.episode_state import EpisodeState
from app.state.graph_builder import compile_episode_graph
from app.services.file_lineage_tracker import FileLineageTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_one_episode")

OUTPUT = Path.cwd() / "output"
OUTPUT.mkdir(exist_ok=True)
for sub in ["images", "videos", "audio", "traces"]:
    (OUTPUT / sub).mkdir(exist_ok=True)

SERIES = "s_test_001"
EP_NUM = 1
EP_ID = f"{SERIES}_ep_{EP_NUM}"

CREATIVE_BRIEF = {
    "theme": "修仙逆袭",
    "genre": "玄幻热血",
    "tone": "热血激昂",
    "total_episodes": 30,
    "target_audience": "18-35岁男性",
    "style": "anime_comic",
    "summary": "少年林风天生废脉，被视为宗门耻辱。一日意外获得上古龙魂传承，从此逆天改命。从被逐出宗门到横扫九州，他誓要以凡人之躯比肩神明。",
    "characters": [
        {"name": "林风", "role": "男主", "traits": ["坚韧", "不屈", "重情义"]},
        {"name": "苏月", "role": "女主", "traits": ["冷傲", "聪慧", "外冷内热"]},
        {"name": "墨渊", "role": "反派", "traits": ["阴险", "野心", "深不可测"]},
    ],
    "core_premise": ["废脉少年获得龙魂传承", "从被逐出宗门到横扫九州", "以凡人之躯比肩神明"],
}

def save_json(name, data):
    p = str(OUTPUT / f"{name}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return p

async def main():
    total_start = time.time()
    trace_id = f"dev_one_{SERIES}_{EP_NUM}"
    tracer = FileLineageTracker(trace_id=trace_id)

    print("=" * 60)
    print("AI-Manga Drama V5: 1 Episode Pipeline (via LangGraph)")
    print("=" * 60)
    print(f"Series: {SERIES} | Episode: {EP_NUM}")
    print(f"Theme: {CREATIVE_BRIEF['theme']} | Genre: {CREATIVE_BRIEF['genre']}")
    print()

    # Build initial state
    state = EpisodeState(
        episode_id=EP_ID,
        series_id=SERIES,
        episode_num=EP_NUM,
        trace_id=trace_id,
        creative_brief=CREATIVE_BRIEF,
    )

    # Compile and stream
    graph = compile_episode_graph()
    config = {"configurable": {"thread_id": EP_ID}}

    print("Running pipeline (streaming)...")
    print("-" * 60)

    final_state = None
    async for event in graph.astream_events(state, config, version="v2"):
        kind = event.get("event", "")
        name = event.get("name", "")

        if kind == "on_chain_start" and name and name not in ("LangGraph", "__start__", "RunnableSequence"):
            print(f"  [{name}] started...")
            tracer.record_node_start(name, event.get("run_id", ""))

        elif kind == "on_chain_end" and name and name not in ("LangGraph", "__start__", "RunnableSequence"):
            output = event.get("data", {}).get("output", {})
            if hasattr(output, "status"):
                print(f"  [{name}] done → status={output.status}")
            elif isinstance(output, dict) and "status" in output:
                print(f"  [{name}] done → status={output['status']}")
            else:
                print(f"  [{name}] done")
            tracer.record_node_end(name, event.get("run_id", ""))

        elif kind == "on_custom_event":
            evt_name = event.get("name", "custom")
            print(f"  [event] {evt_name}")

        # Capture final state
        if kind == "on_chain_end" and name == "LangGraph":
            output = event.get("data", {}).get("output", {})
            if output:
                final_state = output

    tracer.flush()
    print("-" * 60)

    if final_state is None:
        print("ERROR: No final state produced")
        return

    # Handle interrupt
    if final_state.status in ("awaiting_creative_gate", "awaiting_quality_review"):
        print(f"PIPELINE INTERRUPTED: {final_state.status}")
        print(f"  thread_id: {EP_ID}")
        print(f"  Resume: graph.ainvoke(Command(resume={{...}}), config)")
        print(f"  Trace: {tracer.filepath}")
        return

    # Print results
    total_dt = time.time() - total_start
    print()
    print("=" * 60)
    print(f"DONE in {total_dt:.0f}s | cost=${final_state.total_cost_usd:.4f}")
    print(f"  Status: {final_state.status}")
    print(f"  Critic Score: {final_state.critic_score:.2f}")
    print(f"  Decision: {final_state.critic_decision}")
    print(f"  Retries: {final_state.retry_count}")
    script = final_state.script or {}
    scenes = script.get("scenes", [])
    print(f"  Scenes: {len(scenes)} | Shots: {len(final_state.storyboard)}")
    if final_state.episode_asset:
        fp = final_state.episode_asset.get("final_video_path", "")
        if fp:
            print(f"  Final Video: {fp}")
    print(f"  Trace: {tracer.filepath}")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
