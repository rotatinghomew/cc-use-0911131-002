"""特征工程：时点过滤、修订归并、每商户一份的特征与缺失规则。

业务影响说明（随发布版一起冻结）：

* ``log_sales``        月均销售额取自然对数。销售金额右偏（少数高营收商户会拉长量纲），
                        取对数后聚类按相对量级而非绝对差额分组，避免高营收商户独占距离。
* ``repeat_rate``      月均复购率，直接衡量客户粘性，是“成长商户”走访的核心业务信号。
* ``digital_share``    月均线上化占比，反映渠道结构。该字段缺失最多，单独保留缺失指示位。
* ``open_ratio``       平均营业天数 / 当月自然天数，区分稳定经营与偶发停业（如 m004 有 open_days=0）。
* ``sales_trend``      后半段均值 / 前半段均值 - 1，刻画经营方向；与水平量纲无关，避免“增长好
                        但体量小”和“体量大但停滞”被混为一谈。
* ``digital_missing``  线上化占比全缺失指示位。把“未知”当作一种可解释状态，而不是悄悄填 0
                        （0 在业务上表示纯线下，与“没采集”含义不同）。

缺失规则：

* 数值字段（sales / repeat_rate / open_days）按“有值月份”聚合；若整段全部缺失，该特征记为缺失。
* digital_share 不做均值填充：全缺失时用训练集中位数补位并置 ``digital_missing=1``，
  使这类商户在主方向上保持中立，同时保留可识别的缺失标记。
* 有效观测月份少于 ``MIN_VALID_MONTHS`` 的商户不进入训练（样本太薄，不足以代表经营形态），
  在 diagnostics 中列名说明。
"""

from __future__ import annotations

import calendar
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

# 冻结的特征顺序：发布后不得增删或重排，score 端按同一顺序取值。
FEATURE_ORDER: tuple[str, ...] = (
    "log_sales",
    "repeat_rate",
    "digital_share",
    "open_ratio",
    "sales_trend",
    "digital_missing",
)

# 真正参与距离计算的聚类轴。digital_missing / open_ratio / sales_trend 仍在
# FEATURE_ORDER 中计算、冻结并写入解释，但不参与聚类，原因：
# * digital_missing 是数据可得性状态而非经营形态，拿它当距离轴会把“没填线上化”
#   的商户单独切成一群（实测会制造一个纯缺失群），与业务分层无关；
# * open_ratio / sales_trend 噪声大（单月停业、季节性就会剧烈摆动），作为辅助
#   描述保留在画像与解释中，但不主导归群，避免把一次歇业误报成经营恶化。
CLUSTER_FEATURES: tuple[str, ...] = (
    "log_sales",
    "repeat_rate",
    "digital_share",
)

# 仅聚类轴超范围才把成员标记为 out_of_support；辅助特征超范围只在解释中提示。
SUPPORT_FEATURES: tuple[str, ...] = CLUSTER_FEATURES

# 缺失严重判定：核心特征（销售、复购）任一全缺，或有效月份不足，即拒绝自动归群。
CORE_FEATURES: tuple[str, ...] = ("log_sales", "repeat_rate")
MIN_VALID_MONTHS = 3

# score 端快照输入允许的字段（新商户通常只有单月截面值）。
SNAPSHOT_FIELDS: tuple[str, ...] = (
    "sales",
    "repeat_rate",
    "digital_share",
    "open_days",
)


class FeatureError(ValueError):
    """输入数据不满足特征工程的最低要求。"""


def _finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _month_days(month: str) -> int:
    year, month_number = (int(part) for part in month.split("-"))
    return calendar.monthrange(year, month_number)[1]


@dataclass(slots=True)
class MonthlyRecord:
    """同一商户同一月份在 cutoff 时点已知的最高修订版本。"""

    merchant_id: str
    month: str
    sales: float | None
    repeat_rate: float | None
    digital_share: float | None
    open_days: float | None


@dataclass(slots=True)
class MerchantFeatures:
    """一个商户在训练中占一份的特征记录。"""

    merchant_id: str
    values: dict[str, float]
    missing: tuple[str, ...]
    observed_months: int
    eligible: bool
    exclusion_reason: str | None = None
    # 原始尺度的均值，用于输出可读的群画像（元/比例/天数）。
    raw_means: dict[str, float | None] = field(default_factory=dict)


def select_known_observations(
    observations: Iterable[Mapping[str, Any]], cutoff: str
) -> list[Mapping[str, Any]]:
    """只保留 cutoff 当天已经入库的观测，杜绝未来信息泄漏。

    同一 (merchant_id, month) 存在多个修订版本时取最高 revision；
    修订号相同则保留排序靠前者并在外部保证确定性（调用方已排序）。
    """
    filtered = [
        row
        for row in observations
        if isinstance(row, Mapping) and str(row.get("known_at", "")) <= cutoff
    ]
    # 全序排序：输入重排不影响结果。
    filtered.sort(
        key=lambda row: (
            str(row.get("merchant_id", "")),
            str(row.get("month", "")),
            -int(row.get("revision", 1) or 1),
        )
    )
    latest: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in filtered:
        key = (str(row["merchant_id"]), str(row["month"]))
        if key not in latest:  # 已按 revision 倒序，首个即最高修订
            latest[key] = row
    return list(latest.values())


def _aggregate(merchant_id: str, rows: Sequence[Mapping[str, Any]]) -> MerchantFeatures:
    rows = sorted(rows, key=lambda row: str(row["month"]))
    observed_months = len(rows)

    def averages(field_name: str) -> tuple[float | None, int]:
        vals = [
            converted
            for converted in (_finite_float(row.get(field_name)) for row in rows)
            if converted is not None
        ]
        if not vals:
            return None, 0
        return sum(vals) / len(vals), len(vals)

    sales_mean, sales_n = averages("sales")
    repeat_mean, repeat_n = averages("repeat_rate")
    digital_mean, digital_n = averages("digital_share")
    open_values = [
        (float(row["open_days"]), _month_days(str(row["month"])))
        for row in rows
        if _finite_float(row.get("open_days")) is not None
    ]
    open_ratio = (
        sum(value for value, _ in open_values) / sum(days for _, days in open_values)
        if open_values
        else None
    )

    # 销售趋势：后三分之一月份均值 / 前三分之一月份均值 - 1；月份不足时按可用切分。
    sales_series = [
        (str(row["month"]), _finite_float(row.get("sales")))
        for row in rows
    ]
    sales_series = [(month, value) for month, value in sales_series if value is not None]
    sales_trend: float | None = None
    if len(sales_series) >= 2:
        cut = max(1, len(sales_series) // 3)
        head = [value for _, value in sales_series[:cut]]
        tail = [value for _, value in sales_series[-cut:]]
        head_mean = sum(head) / len(head)
        tail_mean = sum(tail) / len(tail)
        if head_mean > 0:
            sales_trend = tail_mean / head_mean - 1.0

    missing: list[str] = []
    values: dict[str, float] = {}

    if sales_mean is None:
        missing.append("log_sales")
    else:
        values["log_sales"] = math.log1p(sales_mean)
    if repeat_mean is None:
        missing.append("repeat_rate")
    else:
        values["repeat_rate"] = repeat_mean
    if digital_mean is None:
        missing.append("digital_share")
        # 占位，实际补位值在拟合时用训练集中位数写入（见 build_feature_matrix）。
        values["digital_share"] = 0.0
        values["digital_missing"] = 1.0
    else:
        values["digital_share"] = digital_mean
        values["digital_missing"] = 0.0
    if open_ratio is None:
        missing.append("open_ratio")
    else:
        values["open_ratio"] = open_ratio
    if sales_trend is None:
        missing.append("sales_trend")
    else:
        values["sales_trend"] = sales_trend

    eligible = True
    reason: str | None = None
    if observed_months < MIN_VALID_MONTHS:
        eligible = False
        reason = f"有效观测月份 {observed_months} < 最低要求 {MIN_VALID_MONTHS}"
    elif any(name in missing for name in CORE_FEATURES):
        eligible = False
        reason = "核心特征（销售额或复购率）整段缺失"

    return MerchantFeatures(
        merchant_id=merchant_id,
        values=values,
        missing=tuple(missing),
        observed_months=observed_months,
        eligible=eligible,
        exclusion_reason=reason,
        raw_means={
            "sales": sales_mean,
            "repeat_rate": repeat_mean,
            "digital_share": digital_mean,
            "open_days": (
                sum(value for value, _ in open_values) / len(open_values)
                if open_values
                else None
            ),
        },
    )


def build_merchant_features(
    observations: Iterable[Mapping[str, Any]], cutoff: str
) -> list[MerchantFeatures]:
    """把原始观测归并成“每商户一份”的特征记录（含不合格样本及原因）。"""
    known = select_known_observations(observations, cutoff)
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in known:
        grouped.setdefault(str(row["merchant_id"]), []).append(row)
    return [_aggregate(merchant_id, grouped[merchant_id]) for merchant_id in sorted(grouped)]


def features_from_snapshot(record: Mapping[str, Any]) -> MerchantFeatures:
    """把 score 端的单月截面新商户转换成同一套特征。

    截面数据没有时间维度，因此 sales_trend 记为缺失并以 0 占位（表示“未见方向”），
    open_ratio 用 open_days / 当月天数（缺月份信息时按 30 天折算）。
    """
    merchant_id = str(record.get("merchant_id", "")).strip()
    if not merchant_id:
        raise FeatureError("score 记录缺少 merchant_id")

    sales = _finite_float(record.get("sales"))
    repeat = _finite_float(record.get("repeat_rate"))
    digital = _finite_float(record.get("digital_share"))
    open_days = _finite_float(record.get("open_days"))

    values: dict[str, float] = {}
    missing: list[str] = []
    if sales is None:
        missing.append("log_sales")
    else:
        values["log_sales"] = math.log1p(max(sales, 0.0))
    if repeat is None:
        missing.append("repeat_rate")
    else:
        values["repeat_rate"] = repeat
    if digital is None:
        missing.append("digital_share")
        values["digital_share"] = 0.0  # 由冻结的中位数补位
        values["digital_missing"] = 1.0
    else:
        values["digital_share"] = digital
        values["digital_missing"] = 0.0
    if open_days is None:
        missing.append("open_ratio")
    else:
        values["open_ratio"] = min(max(open_days / 30.0, 0.0), 1.0)
    # 截面无法计算趋势：中立占位并登记缺失。
    values["sales_trend"] = 0.0
    missing.append("sales_trend")

    eligible = not any(name in missing for name in CORE_FEATURES)
    reason = None if eligible else "核心特征（销售额或复购率）缺失，无法自动归群"
    return MerchantFeatures(
        merchant_id=merchant_id,
        values=values,
        missing=tuple(missing),
        observed_months=1,
        eligible=eligible,
        exclusion_reason=reason,
        raw_means={
            "sales": sales,
            "repeat_rate": repeat,
            "digital_share": digital,
            "open_days": open_days,
        },
    )
