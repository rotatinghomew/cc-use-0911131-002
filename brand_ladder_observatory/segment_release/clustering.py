"""纯标准库、确定性的聚类候选与质量度量。

提供两种有实质差异的候选方案，而不是“同一分数段换名字”：

1. ``kmeans``        k-means++ 初始化的 Lloyd 划分聚类。假设群为球形、方差相近，
                     产出的中心可直接冻结为归群规则，新商户按最近中心入群。
2. ``ward``          Ward 层次凝聚聚类。最小化群内平方和增量，不依赖随机初始化、
                     不预设球形边界，用于检验 k-means 看到的结构是否真实存在。

两者都只依赖标准库；所有随机性由固定种子的 ``random.Random`` 驱动，
且最终标签按中心排序重编号，因此同一份数据无论输入顺序如何结果一致。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

# 发布门槛：轮廓系数低于该值视为“根本不可分”，拒绝发布。
SILHOUETTE_GATE = 0.35
# 最小群规模（绝对户数）与相对占比：低于任一阈值视为“碎片化”，不予发布。
MIN_CLUSTER_SIZE = 5
MIN_CLUSTER_FRACTION = 0.05
MIN_TRAINING_SAMPLES = 12
# 单标准误差式的简约容忍带：当更小的 k 轮廓系数与最优值差距在该容忍内时，
# 优先取更小的 k，避免为了微弱的轮廓提升而切出缺失值/噪声碎片群。
SILHOUETTE_PARSIMONY_TOLERANCE = 0.03
KMEANS_INIT_COUNT = 10
KMEANS_MAX_ITER = 100


@dataclass(slots=True)
class ClusterResult:
    method: str
    k: int
    labels: tuple[int, ...]
    centers: tuple[tuple[float, ...], ...]
    inertia: float
    silhouette: float
    sizes: tuple[int, ...]


def _squared_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return sum((x - y) * (x - y) for x, y in zip(a, b))


def _mean_vector(rows: Sequence[Sequence[float]]) -> tuple[float, ...]:
    width = len(rows[0])
    return tuple(sum(row[j] for row in rows) / len(rows) for j in range(width))


def kmeans(
    matrix: Sequence[Sequence[float]], k: int, seed: int = 17
) -> ClusterResult:
    """k-means++ 初始化、多次确定性重启，取最小惯量（inertia）。"""
    n = len(matrix)
    if n < k:
        raise ValueError("k 不能大于样本数")
    best_labels: tuple[int, ...] | None = None
    best_centers: tuple[tuple[float, ...], ...] | None = None
    best_inertia = math.inf

    for init_index in range(KMEANS_INIT_COUNT):
        rng = random.Random(seed + init_index * 1009)
        centers = _kmeans_pp_seeds(matrix, k, rng)
        labels = [0] * n
        for _ in range(KMEANS_MAX_ITER):
            new_labels = [_nearest_center(row, centers) for row in matrix]
            if new_labels == labels:
                break
            labels = new_labels
            centers = _refresh_centers(matrix, labels, k, centers)
        inertia = sum(
            _squared_distance(row, centers[labels[i]]) for i, row in enumerate(matrix)
        )
        if inertia < best_inertia - 1e-12:
            best_inertia = inertia
            best_labels = tuple(labels)
            best_centers = tuple(centers)

    assert best_labels is not None and best_centers is not None
    ordered_labels, ordered_centers = _order_clusters(matrix, best_labels, best_centers)
    sizes = tuple(best_labels.count(cluster) for cluster in range(k))
    return ClusterResult(
        method="kmeans_pp",
        k=k,
        labels=ordered_labels,
        centers=ordered_centers,
        inertia=round(best_inertia, 8),
        silhouette=silhouette_score(matrix, ordered_labels),
        sizes=tuple(ordered_labels.count(cluster) for cluster in range(k)),
    )


def _kmeans_pp_seeds(
    matrix: Sequence[Sequence[float]], k: int, rng: random.Random
) -> list[tuple[float, ...]]:
    first = rng.randrange(len(matrix))
    centers = [tuple(matrix[first])]
    closest = [_squared_distance(row, centers[0]) for row in matrix]
    for _ in range(1, k):
        total = sum(closest)
        if total <= 0.0:
            # 退化为等概率抽取，保证确定性且不除零。
            index = rng.randrange(len(matrix))
        else:
            pick = rng.random() * total
            cumulative = 0.0
            index = len(matrix) - 1
            for i, weight in enumerate(closest):
                cumulative += weight
                if cumulative >= pick:
                    index = i
                    break
        centers.append(tuple(matrix[index]))
        closest = [
            min(old, _squared_distance(row, centers[-1]))
            for old, row in zip(closest, matrix)
        ]
    return centers


def _nearest_center(
    row: Sequence[float], centers: Sequence[Sequence[float]]
) -> int:
    return min(
        range(len(centers)),
        key=lambda cluster: (_squared_distance(row, centers[cluster]), cluster),
    )


def _refresh_centers(
    matrix: Sequence[Sequence[float]],
    labels: Sequence[int],
    k: int,
    previous: Sequence[Sequence[float]],
) -> list[tuple[float, ...]]:
    centers: list[tuple[float, ...]] = []
    for cluster in range(k):
        members = [row for i, row in enumerate(matrix) if labels[i] == cluster]
        centers.append(_mean_vector(members) if members else tuple(previous[cluster]))
    return centers


def ward_hierarchical(
    matrix: Sequence[Sequence[float]], k: int
) -> ClusterResult:
    """Ward 层次凝聚：初始每个样本一群，反复合并使群内平方和增量最小的两个群。

    群间相异度采用 Ward 闭式
    ``d(A,B) = |A||B|/(|A|+|B|) * ||center_A-center_B||^2``，
    每轮合并后重算距离矩阵。样本量为百级商户时 O(n^3) 足够快，
    且不依赖任何随机初始化，完全确定性。
    """
    n = len(matrix)
    clusters: list[set[int]] = [{i} for i in range(n)]

    while len(clusters) > k:
        distances = _ward_distance_matrix(clusters, matrix)
        a, b = min(distances, key=lambda key: (distances[key], key))
        merged = clusters[a] | clusters[b]
        clusters = [merged] + [
            cluster for index, cluster in enumerate(clusters) if index not in (a, b)
        ]

    labels = [0] * n
    for cluster_id, members in enumerate(clusters):
        for index in members:
            labels[index] = cluster_id
    centers = tuple(
        _mean_vector([matrix[i] for i in members]) for members in clusters
    )
    ordered_labels, ordered_centers = _order_clusters(
        matrix, tuple(labels), centers
    )
    inertia = sum(
        _squared_distance(row, ordered_centers[ordered_labels[i]])
        for i, row in enumerate(matrix)
    )
    return ClusterResult(
        method="ward_hierarchical",
        k=k,
        labels=ordered_labels,
        centers=ordered_centers,
        inertia=round(inertia, 8),
        silhouette=silhouette_score(matrix, ordered_labels),
        sizes=tuple(ordered_labels.count(cluster) for cluster in range(k)),
    )


def _ward_distance_matrix(
    clusters: Sequence[set[int]], matrix: Sequence[Sequence[float]]
) -> dict[tuple[int, int], float]:
    """Ward 相异度的闭式：d(A,B) = |A||B|/(|A|+|B|) * ||center_A-center_B||^2。"""
    centers = [_mean_vector([matrix[i] for i in members]) for members in clusters]
    result: dict[tuple[int, int], float] = {}
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            scale = len(clusters[i]) * len(clusters[j]) / (
                len(clusters[i]) + len(clusters[j])
            )
            result[(i, j)] = scale * _squared_distance(centers[i], centers[j])
    return result


def _order_clusters(
    matrix: Sequence[Sequence[float]],
    labels: Sequence[int],
    centers: Sequence[Sequence[float]],
) -> tuple[tuple[int, ...], tuple[tuple[float, ...], ...]]:
    """按第一特征（标准化前对应 log_sales 的列）中心升序重排群编号。

    这样群编号在跨版本之间有稳定的业务含义（小号=低体量），也保证
    不同算法、不同输入顺序产出的标签可比较。
    """
    order = sorted(range(len(centers)), key=lambda cluster: (centers[cluster][0], cluster))
    remap = {old: new for new, old in enumerate(order)}
    new_labels = tuple(remap[label] for label in labels)
    new_centers = tuple(tuple(centers[old]) for old in order)
    return new_labels, new_centers


def silhouette_score(
    matrix: Sequence[Sequence[float]], labels: Sequence[int]
) -> float:
    """轮廓系数：a=到同群平均距离，b=到最近异群平均距离，取 (b-a)/max(a,b) 的均值。"""
    n = len(matrix)
    unique = set(labels)
    if n < 2 or len(unique) < 2:
        return 0.0
    scores: list[float] = []
    for i in range(n):
        same = [j for j in range(n) if labels[j] == labels[i] and j != i]
        a_i = (
            sum(math.sqrt(_squared_distance(matrix[i], matrix[j])) for j in same)
            / len(same)
            if same
            else 0.0
        )
        b_i = math.inf
        for cluster in unique:
            if cluster == labels[i]:
                continue
            others = [j for j in range(n) if labels[j] == cluster]
            distance = sum(
                math.sqrt(_squared_distance(matrix[i], matrix[j])) for j in others
            ) / len(others)
            b_i = min(b_i, distance)
        if b_i == math.inf:
            continue
        denominator = max(a_i, b_i)
        scores.append(0.0 if denominator == 0.0 else (b_i - a_i) / denominator)
    return round(sum(scores) / len(scores), 6) if scores else 0.0


def adjusted_rand_index(
    labels_a: Sequence[int], labels_b: Sequence[int]
) -> float:
    """调整兰德指数：度量两套划分的一致程度（剔除随机一致），用于交叉验证候选方案。"""
    n = len(labels_a)
    if n != len(labels_b) or n < 2:
        return 0.0
    contingency: dict[tuple[int, int], int] = {}
    for a, b in zip(labels_a, labels_b):
        contingency[(a, b)] = contingency.get((a, b), 0) + 1

    def comb2(value: int) -> float:
        return value * (value - 1) / 2.0

    sum_pairs = sum(comb2(count) for count in contingency.values())
    rows: dict[int, int] = {}
    cols: dict[int, int] = {}
    for (a, b), count in contingency.items():
        rows[a] = rows.get(a, 0) + count
        cols[b] = cols.get(b, 0) + count
    sum_rows = sum(comb2(count) for count in rows.values())
    sum_cols = sum(comb2(count) for count in cols.values())
    total = comb2(n)
    expected = sum_rows * sum_cols / total if total else 0.0
    max_index = (sum_rows + sum_cols) / 2.0
    denominator = max_index - expected
    if denominator == 0.0:
        return 1.0 if sum_pairs == expected else 0.0
    return round((sum_pairs - expected) / denominator, 6)
