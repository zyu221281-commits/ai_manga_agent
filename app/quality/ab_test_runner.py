"""A/B 测试运行器（视频分级比例验证）

 文档 8.2：
- 同集多版本渲染对比（不同 key_scene_ratio）
- 对比各版本：成本 / 渲染时长 / 完播率
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ABTestVariant:
    variant_id: str
    key_scene_ratio: float
    episode_asset: Optional[dict] = None
    cost_usd: float = 0.0
    render_time_s: float = 0.0
    completion_rate: Optional[float] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ABTestResult:
    episode_id: str
    variants: list[ABTestVariant]
    winner: Optional[str] = None
    recommendation: str = ""


RATIO_VARIANTS = [0.10, 0.20, 0.30, 0.50]


class ABTestRunner:
    """视频分级 A/B 测试运行器。

    用法:
        runner = ABTestRunner()
        result = await runner.run_test("ep_1", script, storyboard, prompts, assets)
        print(f"Winner: ratio={result.winner}")
    """

    def __init__(self):
        self._results: dict[str, ABTestResult] = {}

    async def run_test(
        self,
        episode_id: str,
        script: dict,
        storyboard: list[dict],
        image_prompts: list[dict],
        asset_library: dict,
        variant_ratios: list[float] | None = None,
    ) -> ABTestResult:
        """对同集用不同 video_key_scene_ratio 渲染对比。

        Args:
            episode_id: 剧集 ID
            script: 剧本
            storyboard: 分镜
            image_prompts: 图像提示词
            asset_library: 资产库
            variant_ratios: 测试比例列表
        """
        import time
        from app.agents.composer import ComposerAgent

        ratios = variant_ratios or RATIO_VARIANTS
        variants = []

        for ratio in ratios:
            variant_id = f"{episode_id}_r{int(ratio * 100)}"
            logger.info("A/B test: rendering %s with ratio=%.0f%%", variant_id, ratio * 100)

            start = time.monotonic()
            composer = ComposerAgent()
            result = await composer.execute(
                script=script,
                storyboard=storyboard,
                image_prompts=image_prompts,
                asset_library=asset_library,
                key_scene_ratio=ratio,
            )
            render_time = time.monotonic() - start

            variants.append(ABTestVariant(
                variant_id=variant_id,
                key_scene_ratio=ratio,
                episode_asset=result.data if result.success else None,
                cost_usd=result.cost_usd,
                render_time_s=render_time,
                metadata={
                    "success": result.success,
                    "error": result.error,
                },
            ))

        # 分析对比
        winner, recommendation = self._compare(variants)

        test_result = ABTestResult(
            episode_id=episode_id,
            variants=variants,
            winner=winner,
            recommendation=recommendation,
        )
        self._results[episode_id] = test_result
        return test_result

    def _compare(self, variants: list[ABTestVariant]) -> tuple[Optional[str], str]:
        """对比各版本，推荐最优比例。

        优先级：成本 > 渲染时长 > 完播率（Demo 阶段）
        """
        valid = [v for v in variants if v.episode_asset is not None]
        if not valid:
            return None, "All variants failed"

        # 成本最低的作为推荐
        best = min(valid, key=lambda v: v.cost_usd)

        # 生成推荐建议
        costs = [(v.key_scene_ratio, v.cost_usd) for v in valid]
        recommendation = (
            f"推荐 ratio={best.key_scene_ratio:.0%}（成本 ${best.cost_usd:.2f}）。"
            f"各比例成本: {[f'{r:.0%}→${c:.2f}' for r, c in costs]}"
        )

        return best.variant_id, recommendation

    def get_result(self, episode_id: str) -> Optional[ABTestResult]:
        return self._results.get(episode_id)

    def update_completion_rates(self, episode_id: str, rates: dict[str, float]):
        """接入真实完播率数据后更新。

        Args:
            episode_id: 剧集 ID
            rates: {variant_id: completion_rate}
        """
        result = self._results.get(episode_id)
        if not result:
            return
        for variant in result.variants:
            if variant.variant_id in rates:
                variant.completion_rate = rates[variant.variant_id]

        # 重新比较（含完播率）
        winner, recommendation = self._compare_with_completion(result.variants)
        result.winner = winner
        result.recommendation = recommendation

    def _compare_with_completion(self, variants: list[ABTestVariant]) -> tuple[Optional[str], str]:
        valid = [v for v in variants if v.completion_rate is not None]
        if not valid:
            return self._compare(variants)

        best = max(valid, key=lambda v: v.completion_rate)
        recommendation = f"完播率最优 ratio={best.key_scene_ratio:.0%}（完播率={best.completion_rate:.0%}）"
        return best.variant_id, recommendation
