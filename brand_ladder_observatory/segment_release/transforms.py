"""随发布版冻结的特征变换：中位数补位、标准化与支持范围。

标准化（z-score）而不是 min-max 的业务原因：销售与复购的绝对量纲差异很大，
标准化后每个特征对距离的贡献按“偏离典型商户多少个标准差”计算，群不会被
金额的大数值主导；中位数/均值/标准差全部来自训练集，score 端原样复用，
保证新商户进入的是旧发布版而不是被悄悄重训。
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from brand_ladder_observatory.segment_release.features import (
    CLUSTER_FEATURES,
    FEATURE_ORDER,
    SUPPORT_FEATURES,
    MerchantFeatures,
)


@dataclass(slots=True)
class FrozenTransformer:
    feature_order: tuple[str, ...]
    means: dict[str, float]
    stds: dict[str, float]
    digital_median: float
    # 原始（标准化前）特征的训练支持区间，用于识别超范围新商户。
    support_ranges: dict[str, tuple[float, float]]
    sales_trend_neutral: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_order": list(self.feature_order),
            "means": self.means,
            "stds": self.stds,
            "digital_median": self.digital_median,
            "support_ranges": {
                key: list(value) for key, value in self.support_ranges.items()
            },
            "sales_trend_neutral": self.sales_trend_neutral,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FrozenTransformer":
        return cls(
            feature_order=tuple(payload["feature_order"]),
            means=dict(payload["means"]),
            stds=dict(payload["stds"]),
            digital_median=float(payload["digital_median"]),
            support_ranges={
                key: tuple(value) for key, value in payload["support_ranges"].items()
            },
            sales_trend_neutral=float(payload["sales_trend_neutral"]),
        )

    def apply(self, record: MerchantFeatures) -> tuple[list[float], list[str]]:
        """返回标准化向量与触发的超范围特征名。

        digital_share 缺失时在**标准化后**把该维度置 0（即训练总体的典型位置），
        使缺失商户的距离只由其真实观测到的销售与复购驱动；补位中位数仅用于
        支持范围等可读输出。否则全局中位数会被中高体量商户抬高，把低体量的
        缺失商户错误拉离其真实同群（实测会制造一个纯补位伪群）。
        """
        raw = dict(record.values)
        digital_missing = "digital_share" in record.missing
        if digital_missing:
            raw["digital_share"] = self.digital_median
        if "sales_trend" in record.missing:
            raw["sales_trend"] = self.sales_trend_neutral

        vector: list[float] = []
        out_of_support: list[str] = []
        for name in self.feature_order:
            value = float(raw.get(name, 0.0))
            low, high = self.support_ranges[name]
            # 只有聚类轴超范围才影响成员状态；辅助特征仅在解释中体现。
            # 补位值不参与超范围判定，否则缺失本身会被误判成“超范围”。
            if (
                name in SUPPORT_FEATURES
                and not (name == "digital_share" and digital_missing)
                and (value < low or value > high)
            ):
                out_of_support.append(name)
            std = self.stds[name]
            scaled = (value - self.means[name]) / std if std > 0 else 0.0
            if name == "digital_share" and digital_missing:
                scaled = 0.0
            vector.append(scaled)
        return vector, out_of_support


def fit_transformer(records: Sequence[MerchantFeatures]) -> FrozenTransformer:
    """仅用合格训练样本拟合变换参数。"""
    eligible = [record for record in records if record.eligible]
    columns: dict[str, list[float]] = {name: [] for name in FEATURE_ORDER}
    digital_observed: list[float] = []
    for record in eligible:
        for name in FEATURE_ORDER:
            if name == "digital_share" and "digital_share" in record.missing:
                continue  # 中位数只能来自真实观测，不能用占位 0
            columns[name].append(float(record.values[name]))
        if "digital_share" not in record.missing:
            digital_observed.append(float(record.values["digital_share"]))

    if not digital_observed:
        raise ValueError("训练集中没有任何 digital_share 观测，无法确定补位中位数")

    means = {name: statistics.fmean(values) for name, values in columns.items()}
    stds = {
        name: (statistics.pstdev(values) if len(values) > 1 else 1.0)
        for name, values in columns.items()
    }
    # 零方差列（如 digital_missing 全为 0）给一个安全尺度，apply 端也会兜底。
    stds = {name: (value if value > 1e-12 else 1.0) for name, value in stds.items()}
    support_ranges = {
        name: (min(values), max(values)) for name, values in columns.items()
    }
    return FrozenTransformer(
        feature_order=FEATURE_ORDER,
        means=means,
        stds=stds,
        digital_median=statistics.median(digital_observed),
        support_ranges=support_ranges,
        sales_trend_neutral=0.0,
    )


def build_matrix(
    records: Sequence[MerchantFeatures], transformer: FrozenTransformer
) -> list[list[float]]:
    """把合格记录转成标准化聚类矩阵，只保留 CLUSTER_FEATURES 对应的列。"""
    cluster_index = [
        transformer.feature_order.index(name) for name in CLUSTER_FEATURES
    ]
    matrix: list[list[float]] = []
    for record in records:
        if not record.eligible:
            continue
        vector, _ = transformer.apply(record)
        matrix.append([vector[i] for i in cluster_index])
    return matrix
