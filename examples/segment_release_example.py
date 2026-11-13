#!/usr/bin/env python3
"""本地端到端例子：两次拟合 + 新客归群，验证发布版可沿用、不被覆盖。

运行（仓库根目录）::

    python3 examples/segment_release_example.py

脚本完全使用标准库、合成数据与临时 SQLite，不访问网络。它演示四件事：

1. 第一次拟合发布 release-1，输入被故意打乱顺序后重跑——release_id 与
   每个商户的归群完全不变（重排输入不会制造“经营迁移”）；
2. 第二次拟合构造了真实的成员变化：一个群拆分、两个旧群合并、一个群消失、
   一个新群出现——transitions 按共有商户续接业务群标识，不虚构一对一对应；
3. 用冻结的 release-1 给新商户 score：正常新客入群、超支持范围与缺失严重者
   单独标记，且全程不重训；
4. release-2 发布后，release-1 的冻结规则与成员仍然可用，旧版不被新版覆盖。
"""

from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path

# 允许从仓库根目录直接以脚本方式运行（examples/ 不是包）。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brand_ladder_observatory.segment_release.pipeline import (
    fit_release,
    score_records,
)
from brand_ladder_observatory.segment_release.storage import SegmentStore


def _observation(merchant_id: str, month: str, sales: float, repeat: float,
                 digital: float | None, open_days: int, known_at: str,
                 revision: int = 1) -> dict:
    return {
        "merchant_id": merchant_id,
        "month": month,
        "revision": revision,
        "known_at": known_at,
        "sales": sales,
        "repeat_rate": repeat,
        "digital_share": digital,
        "open_days": open_days,
    }


def _merchant_block(prefix: str, index: int, tier: int, months: list[str],
                    cutoff_day: str = "28") -> list[dict]:
    """生成一个商户 6 个月的观测；tier 0/1/2 对应低/中/高三个经营层级。"""
    base_sales = (950.0, 2000.0, 3000.0)[tier]
    base_repeat = (0.15, 0.35, 0.55)[tier]
    base_digital = (0.05, 0.35, 0.65)[tier]
    merchant_id = f"{prefix}{index:03d}"
    rows = []
    for offset, month in enumerate(months):
        wobble = 1.0 + 0.02 * ((index + offset) % 3 - 1)
        # 个别商户线上化缺失，验证缺失规则不会把它们单独切成一群。
        digital = None if index % 13 == 0 else base_digital
        rows.append(
            _observation(
                merchant_id,
                month,
                round(base_sales * wobble, 2),
                round(base_repeat + 0.005 * (offset - 2), 4),
                digital,
                26,
                f"{month}-{cutoff_day}",
            )
        )
    return rows


def _period_payload(cutoff: str, months: list[str], spec: dict,
                    previous_release: str | None = None) -> dict:
    observations: list[dict] = []
    for merchant_id, tier in spec.items():
        prefix = "".join(ch for ch in merchant_id if not ch.isdigit())
        index = int("".join(ch for ch in merchant_id if ch.isdigit()))
        observations.extend(
            _merchant_block(prefix, index, tier, months)
        )
    return {
        "seed": 17,
        "training_cutoff": cutoff,
        "previous_release": previous_release,
        "observations": observations,
    }


def main() -> int:
    temp_dir = tempfile.TemporaryDirectory()
    database = Path(temp_dir.name) / "segment_example.sqlite3"
    store = SegmentStore(database)

    months_1 = [f"2026-{m:02d}" for m in range(1, 7)]
    # release-1：A 群 12 户低层级、B 群 8 户中层级、C 群 8 户高层级。
    spec_1 = {
        **{f"a{i:03d}": 0 for i in range(12)},
        **{f"b{i:03d}": 1 for i in range(8)},
        **{f"c{i:03d}": 2 for i in range(8)},
    }
    payload_1 = _period_payload("2026-06-30", months_1, spec_1)

    print("=" * 72)
    print("步骤 1：第一次拟合（release-1）")
    release_1 = fit_release(payload_1, store=store)
    assert release_1["status"] == "published", release_1["diagnostics"]
    release_1_id = release_1["release_id"]
    for segment in release_1["segments"]:
        print(f"  {segment['segment_id']} {segment['label']} 规模 {segment['size']}")
    print(f"  release_id = {release_1_id}")

    print("=" * 72)
    print("步骤 2：打乱观测顺序后用相同数据重跑")
    shuffled = dict(payload_1)
    shuffled_obs = list(payload_1["observations"])
    random.Random(99).shuffle(shuffled_obs)
    shuffled["observations"] = shuffled_obs
    replay = fit_release(shuffled, store=store)
    same_id = replay["release_id"] == release_1_id
    same_members = (
        {m["merchant_id"]: m["segment_id"] for m in release_1["members"]}
        == {m["merchant_id"]: m["segment_id"] for m in replay["members"]}
    )
    print(f"  release_id 不变：{same_id}（{replay['release_id']}）")
    print(f"  全体商户归群不变：{same_members}")
    assert same_id and same_members, "重排输入制造了迁移，幂等性被破坏"

    print("=" * 72)
    print("步骤 3：第二次拟合，制造真实成员变化（拆分/合并/消失/新生）")
    months_2 = [f"2026-{m:02d}" for m in range(4, 10)]
    # A 群 12 户对半拆分（各 6 户，均高于碎片群下限）：一半留低层、一半升到中层；
    # 升上去的 A 成员与 B 群合并；C 群整批退出样本（群消失）；新增 D 群高层级。
    spec_2: dict = {}
    for i in range(12):
        spec_2[f"a{i:03d}"] = 0 if i < 6 else 1
    for i in range(8):
        spec_2[f"b{i:03d}"] = 1
    for i in range(8):  # 全新的高层级商户群
        spec_2[f"d{i:03d}"] = 2
    payload_2 = _period_payload("2026-09-30", months_2, spec_2)
    release_2 = fit_release(payload_2, store=store)
    assert release_2["status"] == "published", release_2["diagnostics"]
    print(f"  release_id = {release_2['release_id']}（前序 {release_2['transitions']['previous_release_id']}）")
    relations = release_2["transitions"]["relations"]
    for relation in relations:
        print(
            f"  {relation['relation']:>12}: {relation['previous_segment_id']} -> "
            f"{relation['segment_id']} 共有商户 {relation['shared_merchants']} "
            f"流向 {relation['flow']}"
        )
    relation_kinds = {relation["relation"] for relation in relations}
    assert {"continuation", "split", "merge", "disappeared", "new"} <= relation_kinds, relation_kinds
    moved = [t for t in release_2["transitions"]["member_transitions"] if t["kind"] == "moved"]
    left = [t for t in release_2["transitions"]["member_transitions"] if t["kind"] == "left_population"]
    joined = [t for t in release_2["transitions"]["member_transitions"] if t["kind"] == "joined_population"]
    print(f"  成员级：真实迁移 {len(moved)} 户、退出 {len(left)} 户、新进 {len(joined)} 户")

    print("=" * 72)
    print("步骤 4：用冻结的 release-1 给新商户 score（不重训）")
    score_payload = {
        "new_merchants": [
            {"merchant_id": "new-normal", "sales": 2050.0, "repeat_rate": 0.34,
             "digital_share": 0.35, "open_days": 25},
            {"merchant_id": "new-extreme", "sales": 120000.0, "repeat_rate": 0.97,
             "digital_share": 0.99, "open_days": 30},
            {"merchant_id": "new-blank", "sales": None, "repeat_rate": None,
             "digital_share": None, "open_days": 0},
        ]
    }
    scored = score_records(score_payload, release_1_id, store=store)
    for member in scored["members"]:
        print(f"  {member['merchant_id']:>11} -> {member['segment_id']} [{member['status']}]")
    statuses = {member["merchant_id"]: member["status"] for member in scored["members"]}
    assert statuses["new-normal"] == "assigned"
    assert statuses["new-extreme"] == "out_of_support"
    assert statuses["new-blank"] == "insufficient_data"

    print("=" * 72)
    print("步骤 5：确认 release-1 未被 release-2 覆盖，仍可独立评分")
    legacy = store.load_release(release_1_id)
    assert legacy["status"] == "published"
    legacy_scored = score_records(
        {"new_merchants": [score_payload["new_merchants"][0]]},
        release_1_id,
        store=store,
    )
    stored_releases = [item["release_id"] for item in store.list_releases()]
    print(f"  库内发布版：{stored_releases}")
    print(f"  release-1 仍可评分：{legacy_scored['members'][0]['segment_id']}")
    assert stored_releases == [release_1_id, release_2["release_id"]]

    print("=" * 72)
    print("全部断言通过：重排无迁移、续接关系真实、新客规则冻结、旧版本保留。")
    print(f"临时数据库：{database}")
    temp_dir.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
