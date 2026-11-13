"""可发布、可沿用的经营分群（segment release）。

本包与既有梯队功能并存，提供两条命令：

* ``python3 -m brand_ladder_observatory.segment_release fit``  拟合并冻结一个发布版；
* ``python3 -m brand_ladder_observatory.segment_release score`` 用已冻结发布版给新商户归群。

设计约束（对应业务诉求）：

* 拟合只使用 ``training_cutoff`` 当天已经入库（``known_at <= cutoff``）的观测；
* 同一商户同月的多个修订版本只取最高 ``revision``，每个商户在训练中只占一份；
* 特征变换、缺失规则、归群规则随发布版整体冻结，score 不触发重训；
* 超支持范围或缺失严重的样本只做标记，不硬塞进某个群；
* 重新拟合时按两个版本共有的商户续接业务群标识，拆分/合并/消失均显式记录，
  不虚构一对一对应；
* 输入顺序变化不改变结果（全部按 merchant_id / month / revision 排序后处理）。
"""

from __future__ import annotations

__all__ = [
    "FEATURE_ORDER",
    "fit_release",
    "score_records",
    "SegmentStore",
    "SegmentReleaseError",
]

from brand_ladder_observatory.segment_release.features import FEATURE_ORDER
from brand_ladder_observatory.segment_release.pipeline import (
    SegmentReleaseError,
    fit_release,
    score_records,
)
from brand_ladder_observatory.segment_release.storage import SegmentStore
