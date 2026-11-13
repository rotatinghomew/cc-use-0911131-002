from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from brand_ladder_observatory.segment_release.features import (
    build_merchant_features,
    features_from_snapshot,
    select_known_observations,
)
from brand_ladder_observatory.segment_release.pipeline import (
    SegmentReleaseError,
    fit_release,
    score_records,
)
from brand_ladder_observatory.segment_release.storage import SegmentStore

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "fixtures/analysis/input.json"


def load_fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


class PointInTimeAndRevisionTests(unittest.TestCase):
    def test_latest_revision_wins_and_future_known_at_excluded(self):
        observations = [
            {"merchant_id": "x", "month": "2026-01", "revision": 1, "known_at": "2026-01-28",
             "sales": 100.0, "repeat_rate": 0.2, "digital_share": 0.1, "open_days": 26},
            {"merchant_id": "x", "month": "2026-01", "revision": 2, "known_at": "2026-02-05",
             "sales": 140.0, "repeat_rate": 0.2, "digital_share": 0.1, "open_days": 26},
            # cutoff 之后才入库的观测必须被剔除，即使月份落在 cutoff 之前。
            {"merchant_id": "x", "month": "2026-02", "revision": 1, "known_at": "2026-07-02",
             "sales": 9999.0, "repeat_rate": 0.9, "digital_share": 0.9, "open_days": 26},
        ]
        known = select_known_observations(observations, "2026-06-30")
        self.assertEqual(len(known), 1)
        self.assertEqual(known[0]["revision"], 2)
        self.assertEqual(known[0]["sales"], 140.0)

    def test_one_feature_row_per_merchant_independent_of_record_count(self):
        payload = load_fixture()
        records = build_merchant_features(payload["observations"], payload["training_cutoff"])
        merchant_ids = [record.merchant_id for record in records]
        self.assertEqual(len(merchant_ids), len(set(merchant_ids)))
        # 90 个商户，每个 6 个月，聚合后仍是 90 行。
        self.assertEqual(len(records), 90)

    def test_merchant_below_minimum_months_is_excluded_with_reason(self):
        observations = [
            {"merchant_id": "thin", "month": f"2026-0{i}", "revision": 1,
             "known_at": f"2026-0{i}-28", "sales": 1000.0, "repeat_rate": 0.2,
             "digital_share": 0.1, "open_days": 26}
            for i in range(1, 3)  # 仅 2 个月 < MIN_VALID_MONTHS(3)
        ]
        records = build_merchant_features(observations, "2026-06-30")
        self.assertFalse(records[0].eligible)
        self.assertIn("有效观测月份", records[0].exclusion_reason)


class FitContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = SegmentStore(Path(self.temp.name) / "seg.sqlite3")
        self.payload = load_fixture()

    def tearDown(self):
        self.temp.cleanup()

    def test_fixture_publishes_three_interpretable_balanced_segments(self):
        result = fit_release(self.payload, store=self.store)
        self.assertEqual(result["status"], "published")
        self.assertIsNotNone(result["release_id"])
        self.assertEqual(len(result["segments"]), 3)
        for segment in result["segments"]:
            self.assertGreaterEqual(segment["size"], 5)
            self.assertTrue(segment["label"])
            self.assertIn("avg_monthly_sales", segment["profile"])
        # 两个实质不同的候选都参与了比较。
        methods = {c["method"] for c in result["diagnostics"]["candidates"]}
        self.assertEqual(methods, {"kmeans_pp", "ward_hierarchical"})
        # 两种独立算法在选中 k 上完全互相印证。
        self.assertEqual(result["diagnostics"]["cross_method_ari"], 1.0)
        self.assertGreaterEqual(
            result["diagnostics"]["selected_silhouette"], 0.35
        )

    def test_output_contract_fields_present(self):
        result = fit_release(self.payload, store=self.store)
        for key in ("release_id", "status", "segments", "members", "transitions", "diagnostics"):
            self.assertIn(key, result)
        for member in result["members"]:
            self.assertEqual(
                set(member), {"merchant_id", "segment_id", "status", "explanation"}
            )
            self.assertIn("summary", member["explanation"])
            self.assertIn("top_drivers", member["explanation"])

    def test_missing_digital_merchants_stay_in_operating_tiers(self):
        result = fit_release(self.payload, store=self.store)
        assignment = {m["merchant_id"]: m["segment_id"] for m in result["members"]}
        missing_merchants = ["m000", "m013", "m026", "m039", "m052", "m065", "m078"]
        # 7 个缺失户分布在三个经营群，而不是被单独切成“缺失群”。
        self.assertEqual(len({assignment[m] for m in missing_merchants}), 3)

    def test_shuffled_input_reuses_release_and_creates_no_migration(self):
        first = fit_release(self.payload, store=self.store)
        shuffled = dict(self.payload)
        shuffled["observations"] = list(reversed(self.payload["observations"]))
        second = fit_release(shuffled, store=self.store)
        self.assertEqual(first["release_id"], second["release_id"])
        first_map = {m["merchant_id"]: m["segment_id"] for m in first["members"]}
        second_map = {m["merchant_id"]: m["segment_id"] for m in second["members"]}
        self.assertEqual(first_map, second_map)
        with self.store.connect() as connection:
            count = connection.execute(
                "SELECT count(*) FROM segment_releases"
            ).fetchone()[0]
        self.assertEqual(count, 1)


class ScoreContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = SegmentStore(Path(self.temp.name) / "seg.sqlite3")
        self.release = fit_release(load_fixture(), store=self.store)
        self.release_id = self.release["release_id"]

    def tearDown(self):
        self.temp.cleanup()

    def test_score_contract_and_three_member_states(self):
        payload = {
            "new_merchants": [
                {"merchant_id": "ok", "sales": 2000.0, "repeat_rate": 0.35,
                 "digital_share": 0.35, "open_days": 26},
                {"merchant_id": "extreme", "sales": 200000.0, "repeat_rate": 0.98,
                 "digital_share": 1.0, "open_days": 30},
                {"merchant_id": "blank", "sales": None, "repeat_rate": None,
                 "digital_share": None, "open_days": 0},
            ]
        }
        result = score_records(payload, self.release_id, self.store)
        self.assertEqual(result["release_id"], self.release_id)
        self.assertEqual(set(result), {"release_id", "members"})
        by_id = {m["merchant_id"]: m for m in result["members"]}
        self.assertEqual(by_id["ok"]["status"], "assigned")
        self.assertEqual(by_id["extreme"]["status"], "out_of_support")
        # 超范围仍给参考群，但状态必须单独标记，不能当正式成员。
        self.assertIn(by_id["extreme"]["segment_id"], {s["segment_id"] for s in self.release["segments"]})
        self.assertEqual(by_id["blank"]["status"], "insufficient_data")
        self.assertIsNone(by_id["blank"]["segment_id"])
        for member in result["members"]:
            self.assertEqual(
                set(member), {"merchant_id", "segment_id", "status", "explanation"}
            )

    def test_unknown_release_raises(self):
        with self.assertRaises(KeyError):
            score_records({"new_merchants": []}, "rel-does-not-exist", self.store)

    def test_scoring_does_not_mutate_frozen_release(self):
        before = self.store.load_release(self.release_id)
        score_records(
            {"new_merchants": [
                {"merchant_id": "ok", "sales": 1000.0, "repeat_rate": 0.15,
                 "digital_share": 0.05, "open_days": 26}]},
            self.release_id,
            self.store,
        )
        after = self.store.load_release(self.release_id)
        self.assertEqual(before, after)

    def test_observation_format_scoring_matches_snapshot_path(self):
        result = score_records(
            {
                "training_cutoff": "2026-06-30",
                "observations": load_fixture()["observations"],
            },
            self.release_id,
            self.store,
        )
        # 观测格式同样产出每商户一条、字段契约一致。
        self.assertEqual(len(result["members"]), 90)
        for member in result["members"]:
            self.assertEqual(
                set(member), {"merchant_id", "segment_id", "status", "explanation"}
            )


class LineageTests(unittest.TestCase):
    def _payload(self, cutoff: str, spec: dict) -> dict:
        observations = []
        for merchant_id, tier in sorted(spec.items()):
            base_sales = (950.0, 2000.0, 3000.0)[tier]
            for month in range(1, 7):
                observations.append(
                    {"merchant_id": merchant_id, "month": f"2026-{month:02d}",
                     "revision": 1, "known_at": f"2026-{month:02d}-28",
                     "sales": base_sales, "repeat_rate": (0.15, 0.35, 0.55)[tier],
                     "digital_share": (0.05, 0.35, 0.65)[tier], "open_days": 26}
                )
        return {"training_cutoff": cutoff, "observations": observations}

    def test_continuation_keeps_business_identity_through_shared_merchants(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SegmentStore(Path(directory) / "seg.sqlite3")
            spec_1 = {
                **{f"a{i:02d}": 0 for i in range(8)},
                **{f"b{i:02d}": 1 for i in range(8)},
                **{f"c{i:02d}": 2 for i in range(8)},
            }
            first = fit_release(self._payload("2026-06-30", spec_1), store=store)
            # 相同商户、相同结构，仅 cutoff 变化：应全部 continuation，无成员迁移。
            second = fit_release(
                self._payload("2026-09-30", spec_1) | {"previous_release": first["release_id"]},
                store=store,
            )
            self.assertEqual(
                second["transitions"]["previous_release_id"], first["release_id"]
            )
            relations = second["transitions"]["relations"]
            self.assertTrue(
                all(r["relation"] == "continuation" for r in relations)
            )
            self.assertFalse(
                any(t["changed"] for t in second["transitions"]["member_transitions"])
            )

    def test_split_and_merge_and_disappearance_are_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SegmentStore(Path(directory) / "seg.sqlite3")
            spec_1 = {
                **{f"a{i:02d}": 0 for i in range(12)},
                **{f"b{i:02d}": 1 for i in range(8)},
                **{f"c{i:02d}": 2 for i in range(8)},
            }
            first = fit_release(self._payload("2026-06-30", spec_1), store=store)
            spec_2 = {
                **{f"a{i:02d}": (0 if i < 6 else 1) for i in range(12)},
                **{f"b{i:02d}": 1 for i in range(8)},
                **{f"d{i:02d}": 2 for i in range(8)},  # c 群整体消失，d 群新生
            }
            second = fit_release(self._payload("2026-09-30", spec_2), store=store)
            kinds = {r["relation"] for r in second["transitions"]["relations"]}
            self.assertIn("split", kinds)
            self.assertIn("merge", kinds)
            self.assertIn("disappeared", kinds)
            self.assertIn("new", kinds)
            # 拆分关系不伪造一对一目标 segment_id。
            split = next(r for r in second["transitions"]["relations"] if r["relation"] == "split")
            self.assertIsNone(split["segment_id"])
            self.assertGreaterEqual(len(split["flow"]), 2)


class RejectionTests(unittest.TestCase):
    def _merchants(self, n: int, sales: float = 1000.0) -> dict:
        observations = []
        for i in range(n):
            for month in range(1, 7):
                observations.append(
                    {"merchant_id": f"m{i:02d}", "month": f"2026-{month:02d}",
                     "revision": 1, "known_at": f"2026-{month:02d}-28",
                     "sales": sales, "repeat_rate": 0.2,
                     "digital_share": 0.1, "open_days": 26}
                )
        return {"training_cutoff": "2026-06-30", "observations": observations}

    def test_too_few_samples_is_rejected_not_forced_into_clusters(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SegmentStore(Path(directory) / "seg.sqlite3")
            result = fit_release(self._merchants(4), store=store)
            self.assertEqual(result["status"], "rejected")
            self.assertIsNone(result["release_id"])
            self.assertEqual(result["segments"], [])
            self.assertIn("rejection_reason", result["diagnostics"])
            with store.connect() as connection:
                self.assertEqual(
                    connection.execute("SELECT count(*) FROM segment_attempts").fetchone()[0], 1
                )
                # 被拒绝的尝试不得污染正式发布表。
                self.assertEqual(
                    connection.execute("SELECT count(*) FROM segment_releases").fetchone()[0], 0
                )

    def test_unseparable_blob_is_rejected(self):
        # 单团致密数据：商户级均值本身在同一个云团内独立抖动，月度噪声不相关，
        # 聚合后不存在多峰结构。任何 k>=2 的划分轮廓系数都无法过门，
        # 必须明确拒绝而不是硬切群。
        import random

        rng = random.Random(2026)
        observations = []
        for i in range(24):
            mean_sales = 1000.0 * rng.uniform(0.93, 1.07)
            mean_repeat = 0.20 * rng.uniform(0.93, 1.07)
            mean_digital = 0.10 * rng.uniform(0.93, 1.07)
            for month in range(1, 7):
                observations.append(
                    {"merchant_id": f"m{i:02d}", "month": f"2026-{month:02d}",
                     "revision": 1, "known_at": f"2026-{month:02d}-28",
                     "sales": mean_sales * rng.uniform(0.97, 1.03),
                     "repeat_rate": min(1.0, mean_repeat * rng.uniform(0.97, 1.03)),
                     "digital_share": min(1.0, mean_digital * rng.uniform(0.97, 1.03)),
                     "open_days": 26}
                )
        payload = {"training_cutoff": "2026-06-30", "observations": observations}
        with tempfile.TemporaryDirectory() as directory:
            store = SegmentStore(Path(directory) / "seg.sqlite3")
            result = fit_release(payload, store=store)
            self.assertEqual(result["status"], "rejected")
            self.assertIn("轮廓系数", result["diagnostics"]["rejection_reason"])

    def test_missing_cutoff_raises(self):
        with self.assertRaises(SegmentReleaseError):
            fit_release({"observations": []}, store=None)


class StorageTests(unittest.TestCase):
    def test_new_release_does_not_overwrite_previous(self):
        payload = load_fixture()
        with tempfile.TemporaryDirectory() as directory:
            store = SegmentStore(Path(directory) / "seg.sqlite3")
            first = fit_release(payload, store=store)
            second_payload = json.loads(json.dumps(payload))
            second_payload["training_cutoff"] = "2026-07-31"
            second = fit_release(second_payload, store=store)
            self.assertNotEqual(first["release_id"], second["release_id"])
            releases = store.list_releases()
            self.assertEqual(len(releases), 2)
            legacy = store.load_release(first["release_id"])
            self.assertEqual(legacy["release_id"], first["release_id"])
            with store.connect() as connection:
                self.assertEqual(
                    connection.execute("PRAGMA integrity_check").fetchone()[0], "ok"
                )


if __name__ == "__main__":
    unittest.main()
