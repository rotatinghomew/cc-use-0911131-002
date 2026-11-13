"""跨发布版的业务群续接关系。

核心原则：编号会置换，但商户身份（merchant_id）不会。两个版本之间只通过
**共有的商户** 建立关系，不按编号或中心位置虚构一对一映射。

对每对 (旧群, 新群) 统计共有商户数，并按旧群成员的去向比例判定关系：

* ``continuation`` 旧群绝大多数（>=80%）共有成员进入同一个新群——业务标识续接；
* ``split``        旧群成员分散到多个新群，且至少两个新群各占 >=30%；
* ``merge``        多个旧群的成员汇入同一个新群，且至少两个旧群各占 >=30%；
* ``mixed``        同时存在拆分与合并，关系多对多，逐条保留，不压缩成一对一；
* ``disappeared``  旧群在新版中没有任何共有成员（成员全部消失或被剔除）。

新群若不承接任何旧群成员，标记为 ``new``；成员级迁移单独输出，
“重排输入导致的编号变化”在成员级不会产生任何迁移（同一商户仍在同一业务群）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

CONTINUATION_RATIO = 0.80
SPLIT_MERGE_RATIO = 0.30


@dataclass(slots=True)
class LineageResult:
    relations: list[dict[str, Any]]
    member_transitions: list[dict[str, Any]]


def build_lineage(
    previous_members: Mapping[str, str],
    current_members: Mapping[str, str],
    previous_segment_ids: Sequence[str],
    current_segment_ids: Sequence[str],
) -> LineageResult:
    """previous/current_members: merchant_id -> segment_id。"""
    shared = sorted(set(previous_members) & set(current_members))
    disappeared_merchants = sorted(set(previous_members) - set(current_members))
    new_merchants = sorted(set(current_members) - set(previous_members))

    # 旧群 -> {新群: 共有商户数}
    outgoing: dict[str, dict[str, int]] = {seg: {} for seg in previous_segment_ids}
    incoming: dict[str, dict[str, int]] = {seg: {} for seg in current_segment_ids}
    for merchant_id in shared:
        old_seg = previous_members[merchant_id]
        new_seg = current_members[merchant_id]
        outgoing[old_seg][new_seg] = outgoing[old_seg].get(new_seg, 0) + 1
        incoming[new_seg][old_seg] = incoming[new_seg].get(old_seg, 0) + 1

    relations: list[dict[str, Any]] = []
    for old_seg in previous_segment_ids:
        flows = outgoing[old_seg]
        old_total = sum(1 for seg in previous_members.values() if seg == old_seg)
        shared_total = sum(flows.values())
        if shared_total == 0:
            relations.append(
                {
                    "relation": "disappeared",
                    "previous_segment_id": old_seg,
                    "segment_id": None,
                    "shared_merchants": 0,
                    "previous_segment_size": old_total,
                    "flow": {},
                }
            )
            continue
        flow_ratio = {seg: count / shared_total for seg, count in flows.items()}
        dominant_seg, dominant_ratio = max(flow_ratio.items(), key=lambda item: item[1])
        split_targets = [seg for seg, ratio in flow_ratio.items() if ratio >= SPLIT_MERGE_RATIO]

        if dominant_ratio >= CONTINUATION_RATIO:
            relation = "continuation"
        elif len(split_targets) >= 2:
            relation = "split"
        else:
            relation = "mixed"
        relations.append(
            {
                "relation": relation,
                "previous_segment_id": old_seg,
                "segment_id": dominant_seg if relation == "continuation" else None,
                "shared_merchants": shared_total,
                "previous_segment_size": old_total,
                "flow": {seg: flows[seg] for seg in sorted(flows)},
            }
        )

    # 合并关系从新群视角补充（continuation 已覆盖一对一，这里只记录多对一）。
    for new_seg in current_segment_ids:
        sources = incoming[new_seg]
        if len(sources) < 2:
            continue
        received = sum(sources.values())
        strong_sources = [
            old_seg
            for old_seg, count in sources.items()
            if count / received >= SPLIT_MERGE_RATIO
        ]
        if len(strong_sources) >= 2:
            relations.append(
                {
                    "relation": "merge",
                    "previous_segment_id": None,
                    "segment_id": new_seg,
                    "shared_merchants": received,
                    "previous_segment_size": None,
                    "flow": {seg: sources[seg] for seg in sorted(sources)},
                }
            )

    # 不承接任何旧成员的新群 = 新生群。
    for new_seg in current_segment_ids:
        if not incoming[new_seg]:
            relations.append(
                {
                    "relation": "new",
                    "previous_segment_id": None,
                    "segment_id": new_seg,
                    "shared_merchants": 0,
                    "previous_segment_size": None,
                    "flow": {},
                }
            )

    member_transitions: list[dict[str, Any]] = []
    for merchant_id in shared:
        old_seg = previous_members[merchant_id]
        new_seg = current_members[merchant_id]
        member_transitions.append(
            {
                "merchant_id": merchant_id,
                "previous_segment_id": old_seg,
                "segment_id": new_seg,
                "changed": old_seg != new_seg,
                "kind": "moved" if old_seg != new_seg else "retained",
            }
        )
    for merchant_id in disappeared_merchants:
        member_transitions.append(
            {
                "merchant_id": merchant_id,
                "previous_segment_id": previous_members[merchant_id],
                "segment_id": None,
                "changed": True,
                "kind": "left_population",
            }
        )
    for merchant_id in new_merchants:
        member_transitions.append(
            {
                "merchant_id": merchant_id,
                "previous_segment_id": None,
                "segment_id": current_members[merchant_id],
                "changed": True,
                "kind": "joined_population",
            }
        )

    member_transitions.sort(key=lambda item: item["merchant_id"])
    relations.sort(
        key=lambda item: (
            item["relation"],
            item["previous_segment_id"] or "",
            item["segment_id"] or "",
        )
    )
    return LineageResult(relations=relations, member_transitions=member_transitions)
