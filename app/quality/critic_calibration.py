"""Critic 评分校准（ 新增）

提供 20-30 集人工标注样本 + 校准脚本，门禁阈值（0.6/0.8）可配置。
校准流程：Pearson r ≥ 0.6 才启用 Critic 门禁。
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CalibrationSample:
    episode_id: str
    script: str
    llm_score: float
    human_score: float
    dimensions_llm: dict[str, float] = field(default_factory=dict)
    dimensions_human: dict[str, float] = field(default_factory=dict)
    calibration_note: str = ""


@dataclass
class CalibrationReport:
    pearson_r: float
    mean_bias: float  # LLM 评分系统性偏差
    samples_count: int
    threshold_reliable: bool  # r ≥ 0.6
    suggested_pass_threshold: float
    suggested_review_threshold: float
    dimension_deviations: dict[str, float]
    details: list[dict] = field(default_factory=list)


class CriticCalibration:
    """Critic 评分校准器。

    Usage:
        cal = CriticCalibration()
        samples = cal.load_samples("tests/fixtures/critic_calibration_samples.json")
        report = cal.calibrate(samples)
        print(f"Pearson r: {report.pearson_r:.3f}, Reliable: {report.threshold_reliable}")
    """

    def load_samples(self, filepath: str) -> list[CalibrationSample]:
        """从 JSON 文件加载校准样本。"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        samples = []
        items = data if isinstance(data, list) else data.get("samples", [])
        for item in items:
            samples.append(CalibrationSample(
                episode_id=item.get("episode_id", ""),
                script=item.get("script", ""),
                llm_score=float(item.get("llm_score", 0)),
                human_score=float(item.get("human_score", 0)),
                dimensions_llm=item.get("dimensions_llm", {}),
                dimensions_human=item.get("dimensions_human", {}),
                calibration_note=item.get("calibration_note", ""),
            ))
        return samples

    def calibrate(self, samples: list[CalibrationSample]) -> CalibrationReport:
        """执行校准，输出 Pearson r + 偏差报告。

        Args:
            samples: 20-30 集人工标注样本

        Returns:
            校准报告，含建议阈值
        """
        if len(samples) < 10:
            logger.warning("Insufficient samples for calibration (%d < 10)", len(samples))
            return CalibrationReport(
                pearson_r=0.0, mean_bias=0.0, samples_count=len(samples),
                threshold_reliable=False, suggested_pass_threshold=0.8,
                suggested_review_threshold=0.6, dimension_deviations={},
            )

        llm_scores = [s.llm_score for s in samples]
        human_scores = [s.human_score for s in samples]

        # Pearson 相关系数
        r = self._pearson(llm_scores, human_scores)

        # 系统性偏差（LLM 平均分 - 人工平均分）
        mean_bias = sum(llm_scores) / len(llm_scores) - sum(human_scores) / len(human_scores)

        # 判断可靠性
        reliable = r >= 0.6

        # 根据偏差建议阈值
        from app.core.config import settings
        if reliable and abs(mean_bias) > 0.05:
            # 系统性偏差显著，调整阈值
            suggested_pass = max(0.5, min(0.95, settings.CRITIC_PASS_THRESHOLD - mean_bias))
            suggested_review = max(0.3, min(0.9, settings.CRITIC_REVIEW_THRESHOLD - mean_bias))
        else:
            suggested_pass = settings.CRITIC_PASS_THRESHOLD
            suggested_review = settings.CRITIC_REVIEW_THRESHOLD

        # 维度偏差
        dim_deviations = self._calculate_dimension_deviations(samples)

        # 详细记录
        details = []
        for s in samples:
            details.append({
                "episode_id": s.episode_id,
                "llm_score": s.llm_score,
                "human_score": s.human_score,
                "delta": s.llm_score - s.human_score,
            })

        return CalibrationReport(
            pearson_r=r,
            mean_bias=mean_bias,
            samples_count=len(samples),
            threshold_reliable=reliable,
            suggested_pass_threshold=round(suggested_pass, 2),
            suggested_review_threshold=round(suggested_review, 2),
            dimension_deviations=dim_deviations,
            details=details,
        )

    @staticmethod
    def _pearson(x: list[float], y: list[float]) -> float:
        n = len(x)
        if n < 2:
            return 0.0
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
        std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))
        if std_x == 0 or std_y == 0:
            return 0.0
        return cov / (std_x * std_y)

    @staticmethod
    def _calculate_dimension_deviations(samples: list[CalibrationSample]) -> dict[str, float]:
        deviations = {}
        all_dims = set()
        for s in samples:
            all_dims.update(s.dimensions_llm.keys())
            all_dims.update(s.dimensions_human.keys())

        for dim in all_dims:
            llm_vals = [s.dimensions_llm.get(dim, 0) for s in samples if dim in s.dimensions_llm]
            human_vals = [s.dimensions_human.get(dim, 0) for s in samples if dim in s.dimensions_human]
            if llm_vals and human_vals:
                deviations[dim] = sum(llm_vals) / len(llm_vals) - sum(human_vals) / len(human_vals)

        return deviations
