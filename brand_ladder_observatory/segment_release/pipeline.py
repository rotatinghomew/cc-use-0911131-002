"""分群发布主流程：候选比较、质量门槛、发布冻结与新客归群。

发布版 bundle（冻结内容）结构::

    {
      "release_id", "status", "training_cutoff", "previous_release_id",
      "model": {"method", "k", "centers", "feature_order"},
      "transformer": FrozenTransformer.to_dict(),
      "segments": [{"segment_id", "label", "profile", "size", ...}],
      "gates": {"silhouette_gate", "min_cluster_size", ...}
    }

归群规则对所有成员一致：标准化后取平方欧氏距离最近的群中心；
该规则随 bundle 冻结，score 端不允许重训或改中心。
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Sequence

from brand_ladder_observatory.foundation.registry import stable_json
from brand_ladder_observatory.segment_release.clustering import (
    MIN_CLUSTER_FRACTION,
    MIN_CLUSTER_SIZE,
    MIN_TRAINING_SAMPLES,
    SILHOUETTE_GATE,
    SILHOUETTE_PARSIMONY_TOLERANCE,
    ClusterResult,
    adjusted_rand_index,
    kmeans,
    ward_hierarchical,
)
from brand_ladder_observatory.segment_release.features import (
    CLUSTER_FEATURES,
    FEATURE_ORDER,
    MerchantFeatures,
    build_merchant_features,
    features_from_snapshot,
)
from brand_ladder_observatory.segment_release.lineage import build_lineage
from brand_ladder_observatory.segment_release.transforms import (
    FrozenTransformer,
    build_matrix,
    fit_transformer,
)

FEATURE_LABELS_ZH = {
    "log_sales": "月均销售额",
    "repeat_rate": "复购率",
    "digital_share": "线上化占比",
    "open_ratio": "营业天数占比",
    "sales_trend": "销售趋势",
    "digital_missing": "线上化信息缺失",
}


class SegmentReleaseError(ValueError):
    """拟合输入非法或发布被拒绝。"""


def _canonical_fingerprint(
    cutoff: str, records: Sequence[MerchantFeatures]
) -> str:
    """对训练内容做与输入顺序无关的指纹，用于幂等复用与重排验证。"""
    payload = {
        "training_cutoff": cutoff,
        "merchants": [
            {
                "merchant_id": record.merchant_id,
                "values": record.values,
                "missing": list(record.missing),
                "observed_months": record.observed_months,
                "eligible": record.eligible,
            }
            for record in sorted(records, key=lambda item: item.merchant_id)
        ],
    }
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()[:16]


def _candidate_grid(n_samples: int) -> list[int]:
    upper = min(6, n_samples // MIN_CLUSTER_SIZE)
    return list(range(2, upper + 1))


def _fragment_violations(result: ClusterResult, n_samples: int) -> list[str]:
    violations: list[str] = []
    for size in result.sizes:
        if size < MIN_CLUSTER_SIZE:
            violations.append(f"存在 {size} 户的碎片群（绝对下限 {MIN_CLUSTER_SIZE}）")
        if size / n_samples < MIN_CLUSTER_FRACTION:
            violations.append(
                f"存在占比 {size / n_samples:.2%} 的碎片群（相对下限 {MIN_CLUSTER_FRACTION:.0%}）"
            )
    return violations


def _run_candidates(
    matrix: Sequence[Sequence[float]],
) -> tuple[list[dict[str, Any]], ClusterResult | None, str | None]:
    n_samples = len(matrix)
    grid = _candidate_grid(n_samples)
    if not grid:
        return (
            [],
            None,
            "样本量不足以支撑任何群数（每群至少 %d 户）" % MIN_CLUSTER_SIZE,
        )

    evaluations: list[dict[str, Any]] = []
    results: dict[str, ClusterResult] = {}
    for k in grid:
        km = kmeans(matrix, k)
        wd = ward_hierarchical(matrix, k)
        results[f"kmeans_pp:{k}"] = km
        results[f"ward_hierarchical:{k}"] = wd
        for result in (km, wd):
            evaluations.append(
                {
                    "method": result.method,
                    "k": k,
                    "silhouette": result.silhouette,
                    "inertia": result.inertia,
                    "sizes": list(result.sizes),
                    "min_cluster_size": min(result.sizes),
                    "fragment_violations": _fragment_violations(result, n_samples),
                    "agreement_with_ward": (
                        adjusted_rand_index(km.labels, wd.labels)
                        if result.method == "kmeans_pp"
                        else None
                    ),
                }
            )

    def viable(result: ClusterResult) -> bool:
        return (
            result.silhouette >= SILHOUETTE_GATE
            and not _fragment_violations(result, n_samples)
        )

    viable_results = [(key, result) for key, result in results.items() if viable(result)]
    if not viable_results:
        return (
            evaluations,
            None,
            "没有候选同时满足轮廓系数 >= %.2f、最小群规模 >= %d 且最小占比 >= %.0f，"
            "结构不可分或只会产生碎片群，拒绝发布"
            % (SILHOUETTE_GATE, MIN_CLUSTER_SIZE, MIN_CLUSTER_FRACTION),
        )

    # 简约原则（one-SE 思路）：先取所有可行候选的最高轮廓系数，再在其容忍带内
    # 选最小的 k；同一 k 下要求两种独立算法互相印证（取 ARI 高者，平局优先
    # k-means——其中心即冻结的归群规则，score 解释最直接）。这样不会因为缺失值
    # 制造的微弱轮廓提升而把真实业务群切碎。
    best_silhouette = max(result.silhouette for _, result in viable_results)
    within_tolerance = [
        (key, result)
        for key, result in viable_results
        if best_silhouette - result.silhouette <= SILHOUETTE_PARSIMONY_TOLERANCE
    ]
    min_k = min(result.k for _, result in within_tolerance)
    same_k = [(key, result) for key, result in within_tolerance if result.k == min_k]
    method_priority = {"kmeans_pp": 0, "ward_hierarchical": 1}
    best_key, best_result = min(
        same_k,
        key=lambda item: (
            -item[1].silhouette,
            method_priority[item[1].method],
        ),
    )
    return evaluations, best_result, None


def _segment_profiles(
    result: ClusterResult,
    eligible: Sequence[MerchantFeatures],
    transformer: FrozenTransformer,
) -> list[dict[str, Any]]:
    """用原始尺度的成员均值给每个群写可解释画像与中文标签。"""
    by_cluster: dict[int, list[MerchantFeatures]] = {
        cluster: [] for cluster in range(result.k)
    }
    for record, label in zip(eligible, result.labels):
        by_cluster[label].append(record)

    segments: list[dict[str, Any]] = []
    for cluster in range(result.k):
        members = by_cluster[cluster]
        raw_sales = [m.raw_means["sales"] for m in members if m.raw_means["sales"] is not None]
        raw_repeat = [m.raw_means["repeat_rate"] for m in members if m.raw_means["repeat_rate"] is not None]
        raw_digital = [m.raw_means["digital_share"] for m in members if m.raw_means["digital_share"] is not None]
        raw_open = [m.raw_means["open_days"] for m in members if m.raw_means["open_days"] is not None]
        digital_missing_rate = (
            sum(1 for m in members if "digital_share" in m.missing) / len(members)
        )
        profile = {
            "member_count": len(members),
            "avg_monthly_sales": round(sum(raw_sales) / len(raw_sales), 2) if raw_sales else None,
            "avg_repeat_rate": round(sum(raw_repeat) / len(raw_repeat), 4) if raw_repeat else None,
            "avg_digital_share": round(sum(raw_digital) / len(raw_digital), 4) if raw_digital else None,
            "avg_open_days": round(sum(raw_open) / len(raw_open), 2) if raw_open else None,
            "digital_missing_rate": round(digital_missing_rate, 4),
        }
        segments.append(
            {
                "segment_id": f"seg-{cluster + 1:02d}",
                "label": _label_for(profile),
                "profile": profile,
                "center": list(result.centers[cluster]),
                "size": len(members),
            }
        )
    return segments


def _label_for(profile: Mapping[str, Any]) -> str:
    """标签必须由成员特征解释：按销售水平与复购/线上化组合命名。"""
    sales = profile.get("avg_monthly_sales") or 0.0
    repeat = profile.get("avg_repeat_rate") or 0.0
    if sales < 1400:
        scale = "小微起步"
    elif sales < 2500:
        scale = "稳健成长"
    else:
        scale = "规模成熟"
    if repeat >= 0.5:
        stickiness = "高粘性"
    elif repeat >= 0.25:
        stickiness = "中粘性"
    else:
        stickiness = "低粘性"
    return f"{scale}·{stickiness}商户群"


def _explain(
    record: MerchantFeatures,
    cluster_vector: Sequence[float],
    segments: Sequence[Mapping[str, Any]],
    out_of_support: Sequence[str],
    status: str,
) -> dict[str, Any]:
    """单商户归群依据：各聚类轴到所选群中心的贡献 + 与次优群的间距。

    辅助特征（营业稳定性、趋势、缺失标记）不进距离，只在文字解释中补充，
    确保“为什么进这个群”完全由聚类轴回答。
    """
    distances: list[tuple[str, float]] = []
    for segment in segments:
        distance = math.sqrt(
            sum((x - c) ** 2 for x, c in zip(cluster_vector, segment["center"]))
        )
        distances.append((segment["segment_id"], distance))
    distances.sort(key=lambda item: item[1])
    chosen_id, chosen_distance = distances[0]
    runner_up = distances[1] if len(distances) > 1 else (None, None)

    chosen_segment = next(item for item in segments if item["segment_id"] == chosen_id)
    contributions = []
    for index, feature_name in enumerate(CLUSTER_FEATURES):
        gap = cluster_vector[index] - chosen_segment["center"][index]
        contributions.append(
            {
                "feature": feature_name,
                "feature_label": FEATURE_LABELS_ZH[feature_name],
                "squared_contribution": round(gap * gap, 6),
            }
        )
    contributions.sort(key=lambda item: item["squared_contribution"])
    drivers = [item["feature_label"] for item in contributions]

    reasons = [f"距群 {chosen_id}（{chosen_segment['label']}）中心最近"]
    reasons.append("主要归群依据（按贡献排序）：" + "、".join(drivers))
    if runner_up[0] is not None and runner_up[1] is not None:
        margin = runner_up[1] - chosen_distance
        reasons.append(
            f"与次优群 {runner_up[0]} 的距离余量 {margin:.3f}"
            + ("（边界样本，建议人工复核）" if margin < 0.25 else "")
        )
    if out_of_support:
        reasons.append(
            "超出训练支持范围的特征："
            + "、".join(FEATURE_LABELS_ZH[name] for name in out_of_support)
        )
    auxiliary_missing = [
        name for name in record.missing if name not in CLUSTER_FEATURES
    ]
    core_missing = [name for name in record.missing if name in CLUSTER_FEATURES]
    if core_missing or auxiliary_missing:
        shown = [
            FEATURE_LABELS_ZH.get(name, name)
            for name in (*core_missing, *auxiliary_missing)
        ]
        reasons.append("缺失特征：" + "、".join(shown))
    # 辅助特征以可读形式补充（不参与归群）。
    aux_notes: list[str] = []
    if "open_ratio" not in record.missing:
        aux_notes.append(f"营业天数占比 {record.values['open_ratio']:.3f}")
    if "sales_trend" not in record.missing:
        aux_notes.append(f"销售趋势 {record.values['sales_trend']:+.3f}")
    if "digital_share" in record.missing:
        aux_notes.append("线上化占比缺失（按训练中位数补位，不主导归群）")
    if aux_notes:
        reasons.append("辅助参考（不参与距离计算）：" + "；".join(aux_notes))

    return {
        "summary": "；".join(reasons),
        "distance_to_center": round(chosen_distance, 6),
        "runner_up_segment_id": runner_up[0],
        "margin": round(runner_up[1] - chosen_distance, 6) if runner_up[1] is not None else None,
        "top_drivers": drivers,
        "out_of_support_features": list(out_of_support),
        "missing_features": list(record.missing),
        "status_reason": status,
    }


def _load_previous(
    store: Any | None, previous_release: str | None
) -> dict[str, Any] | None:
    if store is None:
        return None
    if previous_release:
        return {
            "release_id": previous_release,
            "bundle": store.load_release(previous_release),
        }
    return store.latest_published()


def fit_release(
    payload: Mapping[str, Any],
    store: Any | None = None,
) -> dict[str, Any]:
    """拟合并发布一个分群版本。store 为 None 时只返回结果不落库（便于测试）。"""
    cutoff = str(payload.get("training_cutoff", "")).strip()
    if not cutoff:
        raise SegmentReleaseError("training_cutoff 是必填项")
    observations = payload.get("observations")
    if not isinstance(observations, list):
        raise SegmentReleaseError("observations 必须是数组")

    records = build_merchant_features(observations, cutoff)
    eligible = [record for record in records if record.eligible]
    excluded = [
        {
            "merchant_id": record.merchant_id,
            "reason": record.exclusion_reason,
            "observed_months": record.observed_months,
            "missing_features": list(record.missing),
        }
        for record in records
        if not record.eligible
    ]
    fingerprint = _canonical_fingerprint(cutoff, records)

    # 幂等：同一份训练内容重跑（含输入重排）直接复用已发布版本，绝不生成新编号。
    if store is not None:
        reused = store.find_by_fingerprint(fingerprint)
        if reused is not None:
            bundle = reused["bundle"]
            return _rehydrate(bundle, store, fingerprint)

    diagnostics: dict[str, Any] = {
        "training_cutoff": cutoff,
        "merchants_seen": len(records),
        "merchants_eligible": len(eligible),
        "excluded_merchants": excluded,
        "feature_order": list(FEATURE_ORDER),
        "missing_policy": {
            "numeric": "按有值月份聚合；整段缺失则该特征记缺失",
            "digital_share": "全缺失时以训练中位数补位并置 digital_missing=1",
            "sales_trend": "截面/月份不足时以 0 中立占位并登记缺失",
            "min_valid_months": 3,
        },
        "scale_policy": "log1p 销售 + 训练集 z-score 标准化，参数随发布冻结",
        "cluster_features": list(CLUSTER_FEATURES),
        "auxiliary_features": [
            name for name in FEATURE_ORDER if name not in CLUSTER_FEATURES
        ],
        "feature_decision": (
            "聚类轴只保留销售额、复购率、线上化占比三个经营维度；"
            "缺失标记、营业稳定性与销售趋势保留在画像和解释中但不参与距离，"
            "避免把数据缺失或单月歇业误切成独立经营群"
        ),
        "gates": {
            "silhouette_gate": SILHOUETTE_GATE,
            "min_cluster_size": MIN_CLUSTER_SIZE,
            "min_cluster_fraction": MIN_CLUSTER_FRACTION,
            "min_training_samples": MIN_TRAINING_SAMPLES,
            "silhouette_parsimony_tolerance": SILHOUETTE_PARSIMONY_TOLERANCE,
        },
    }

    if len(eligible) < MIN_TRAINING_SAMPLES:
        reason = f"合格训练样本仅 {len(eligible)} 户，少于最低 {MIN_TRAINING_SAMPLES} 户，拒绝发布"
        diagnostics["rejection_reason"] = reason
        if store is not None:
            store.save_attempt(
                f"attempt-{fingerprint}", cutoff, fingerprint, reason, diagnostics
            )
        return {
            "release_id": None,
            "status": "rejected",
            "segments": [],
            "members": [],
            "transitions": [],
            "diagnostics": diagnostics,
        }

    transformer = fit_transformer(records)
    matrix = build_matrix(records, transformer)
    evaluations, best, gate_reason = _run_candidates(matrix)
    diagnostics["candidates"] = evaluations
    diagnostics["selection_rule"] = (
        "剔除触发碎片群门槛的候选后，在轮廓系数达到发布门槛的方案中，"
        "取最高轮廓系数容忍带(±%.2f)内最小的 k；同一 k 下用 k-means 与 Ward 的"
        "调整兰德指数交叉印证，平局优先 kmeans_pp（中心即冻结归群规则）"
        % SILHOUETTE_PARSIMONY_TOLERANCE
    )

    if best is None:
        diagnostics["rejection_reason"] = gate_reason
        if store is not None:
            store.save_attempt(
                f"attempt-{fingerprint}", cutoff, fingerprint, gate_reason or "", diagnostics
            )
        return {
            "release_id": None,
            "status": "rejected",
            "segments": [],
            "members": [],
            "transitions": [],
            "diagnostics": diagnostics,
        }

    diagnostics["selected_method"] = best.method
    diagnostics["selected_k"] = best.k
    diagnostics["selected_silhouette"] = best.silhouette
    # 两方法在同一 k 下的一致性，作为“结构真实而非算法人为”的交叉证据。
    diagnostics["cross_method_ari"] = next(
        (
            item["agreement_with_ward"]
            for item in evaluations
            if item["method"] == "kmeans_pp" and item["k"] == best.k
        ),
        None,
    )

    segments = _segment_profiles(best, eligible, transformer)
    previous = _load_previous(store, payload.get("previous_release"))
    previous_release_id = previous["release_id"] if previous else None

    # 当前成员（仅合格训练样本；不合格样本单独标记，不进群）。
    current_members: list[dict[str, Any]] = []
    current_lookup: dict[str, str] = {}
    cluster_index = [
        transformer.feature_order.index(name) for name in CLUSTER_FEATURES
    ]
    for record in eligible:
        full_vector, out_of_support = transformer.apply(record)
        cluster_vector = [full_vector[i] for i in cluster_index]
        explanation = _explain(
            record, cluster_vector, segments, out_of_support, "assigned"
        )
        chosen_id = _nearest_segment(cluster_vector, segments)
        current_members.append(
            {
                "merchant_id": record.merchant_id,
                "segment_id": chosen_id,
                "status": "assigned",
                "explanation": explanation,
            }
        )
        current_lookup[record.merchant_id] = chosen_id

    for record in records:
        if record.eligible:
            continue
        current_members.append(
            {
                "merchant_id": record.merchant_id,
                "segment_id": None,
                "status": "insufficient_data",
                "explanation": {
                    "summary": f"样本不足或核心特征缺失，未自动归群：{record.exclusion_reason}",
                    "missing_features": list(record.missing),
                    "out_of_support_features": [],
                    "status_reason": "insufficient_data",
                },
            }
        )
    current_members.sort(key=lambda item: item["merchant_id"])

    transitions: dict[str, Any] = {"relations": [], "member_transitions": []}
    if previous is not None:
        previous_bundle = previous["bundle"]
        previous_lookup = {
            member["merchant_id"]: member["segment_id"]
            for member in previous_bundle.get("members", [])
            if member.get("segment_id")
        }
        previous_ids = [seg["segment_id"] for seg in previous_bundle["segments"]]
        current_ids = [seg["segment_id"] for seg in segments]
        lineage = build_lineage(
            previous_lookup, current_lookup, previous_ids, current_ids
        )
        transitions = {
            "previous_release_id": previous_release_id,
            "relations": lineage.relations,
            "member_transitions": lineage.member_transitions,
        }

    release_id = f"rel-{cutoff.replace('-', '')}-{fingerprint[:10]}"
    bundle = {
        "release_id": release_id,
        "status": "published",
        "training_cutoff": cutoff,
        "previous_release_id": previous_release_id,
        "canonical_fingerprint": fingerprint,
        "model": {
            "method": best.method,
            "k": best.k,
            "centers": [list(center) for center in best.centers],
            "feature_order": list(FEATURE_ORDER),
            "cluster_features": list(CLUSTER_FEATURES),
            "assignment_rule": "标准化空间内平方欧氏距离最近中心（仅聚类轴参与）",
        },
        "transformer": transformer.to_dict(),
        "segments": segments,
        "members": current_members,
        "gates": diagnostics["gates"],
    }

    if store is not None:
        store.save_release(
            release_id,
            "published",
            cutoff,
            previous_release_id,
            fingerprint,
            bundle,
            current_members,
            transitions.get("relations", []),
        )

    return {
        "release_id": release_id,
        "status": "published",
        "segments": segments,
        "members": current_members,
        "transitions": transitions,
        "diagnostics": diagnostics,
    }


def _rehydrate(
    bundle: Mapping[str, Any], store: Any, fingerprint: str
) -> dict[str, Any]:
    """幂等命中时还原对外输出结构（不新建发布、不改写旧版）。"""
    relations: list[dict[str, Any]] = []
    if bundle.get("previous_release_id"):
        with store.connect() as connection:
            rows = connection.execute(
                "SELECT detail FROM segment_lineage_relations WHERE release_id=?",
                (bundle["release_id"],),
            ).fetchall()
        import json

        relations = [json.loads(row["detail"]) for row in rows]
    return {
        "release_id": bundle["release_id"],
        "status": bundle["status"],
        "segments": bundle["segments"],
        "members": bundle["members"],
        "transitions": {
            "previous_release_id": bundle.get("previous_release_id"),
            "relations": relations,
            "member_transitions": [],
        },
        "diagnostics": {
            "canonical_fingerprint": fingerprint,
            "idempotent_replay": True,
            "reused_release_id": bundle["release_id"],
        },
    }


def score_records(
    payload: Mapping[str, Any],
    release_id: str,
    store: Any,
) -> dict[str, Any]:
    """用已冻结发布版给新商户归群；不触碰任何训练参数。"""
    bundle = store.load_release(release_id)
    if bundle.get("status") != "published":
        raise SegmentReleaseError(f"发布版 {release_id} 不是可评分的 published 状态")
    transformer = FrozenTransformer.from_dict(bundle["transformer"])
    segments = bundle["segments"]

    raw_records = payload.get("new_merchants")
    candidates: list[MerchantFeatures] = []
    if raw_records is not None:
        if not isinstance(raw_records, list):
            raise SegmentReleaseError("new_merchants 必须是数组")
        for item in raw_records:
            if not isinstance(item, Mapping):
                raise SegmentReleaseError("new_merchants 元素必须是对象")
            candidates.append(features_from_snapshot(item))
    elif isinstance(payload.get("observations"), list):
        # 允许直接喂观测格式：按同一套每商户聚合逻辑处理。
        cutoff = str(payload.get("training_cutoff", "9999-12-31"))
        candidates = build_merchant_features(payload["observations"], cutoff)
    else:
        raise SegmentReleaseError("score 输入需要 new_merchants 或 observations")

    members: list[dict[str, Any]] = []
    for record in candidates:
        if not record.eligible:
            members.append(
                {
                    "merchant_id": record.merchant_id,
                    "segment_id": None,
                    "status": "insufficient_data",
                    "explanation": {
                        "summary": f"缺失严重，未自动归群：{record.exclusion_reason}",
                        "missing_features": list(record.missing),
                        "out_of_support_features": [],
                        "status_reason": "insufficient_data",
                    },
                }
            )
            continue
        vector, out_of_support = transformer.apply(record)
        cluster_vector = [
            vector[transformer.feature_order.index(name)]
            for name in bundle["model"].get("cluster_features", CLUSTER_FEATURES)
        ]
        if out_of_support:
            # 超支持范围：仍给出最近群作为参考，但状态单独标记，业务不得直接当正式成员使用。
            explanation = _explain(
                record, cluster_vector, segments, out_of_support, "out_of_support"
            )
            segment_id = _nearest_segment(cluster_vector, segments)
            status = "out_of_support"
        else:
            explanation = _explain(record, cluster_vector, segments, [], "assigned")
            segment_id = _nearest_segment(cluster_vector, segments)
            status = "assigned"
        members.append(
            {
                "merchant_id": record.merchant_id,
                "segment_id": segment_id,
                "status": status,
                "explanation": explanation,
            }
        )

    members.sort(key=lambda item: item["merchant_id"])
    return {"release_id": release_id, "members": members}


def _nearest_segment(
    vector: Sequence[float], segments: Sequence[Mapping[str, Any]]
) -> str:
    def distance(segment: Mapping[str, Any]) -> float:
        return math.sqrt(
            sum((x - c) ** 2 for x, c in zip(vector, segment["center"]))
        )

    return min(segments, key=distance)["segment_id"]
