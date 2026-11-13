"""分群发布版的 SQLite 存储，复用既有账本的连接约定。

发布即冻结：``segment_releases.bundle`` 保存完整的特征变换、缺失规则与归群中心，
score 只读取该包，不修改任何已发布行。新版以新的 release_id 插入，
旧版永远不会被覆盖；被拒绝发布的尝试单独记录，便于审计“为什么没有强行出群”。
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence

from brand_ladder_observatory.foundation.registry import stable_json, utc_now


class SegmentStore:
    def __init__(self, database: str | Path):
        self.database = str(database)
        self._initialize()

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.database, timeout=20)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS segment_releases(
                    release_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    training_cutoff TEXT NOT NULL,
                    previous_release_id TEXT,
                    canonical_fingerprint TEXT NOT NULL,
                    bundle TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS segment_release_members(
                    release_id TEXT NOT NULL,
                    merchant_id TEXT NOT NULL,
                    segment_id TEXT,
                    status TEXT NOT NULL,
                    explanation TEXT NOT NULL,
                    PRIMARY KEY(release_id, merchant_id)
                );
                CREATE INDEX IF NOT EXISTS segment_members_release
                    ON segment_release_members(release_id);
                CREATE TABLE IF NOT EXISTS segment_lineage_relations(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    release_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    previous_segment_id TEXT,
                    segment_id TEXT,
                    detail TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS segment_attempts(
                    attempt_id TEXT PRIMARY KEY,
                    training_cutoff TEXT NOT NULL,
                    canonical_fingerprint TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    diagnostics TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def save_release(
        self,
        release_id: str,
        status: str,
        training_cutoff: str,
        previous_release_id: str | None,
        canonical_fingerprint: str,
        bundle: Mapping[str, Any],
        members: Sequence[Mapping[str, Any]],
        relations: Sequence[Mapping[str, Any]],
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO segment_releases VALUES(?,?,?,?,?,?,?)",
                (
                    release_id,
                    status,
                    training_cutoff,
                    previous_release_id,
                    canonical_fingerprint,
                    stable_json(bundle),
                    utc_now(),
                ),
            )
            connection.executemany(
                "INSERT OR REPLACE INTO segment_release_members VALUES(?,?,?,?,?)",
                [
                    (
                        release_id,
                        member["merchant_id"],
                        member.get("segment_id"),
                        member["status"],
                        stable_json(member["explanation"]),
                    )
                    for member in members
                ],
            )
            for relation in relations:
                connection.execute(
                    "INSERT INTO segment_lineage_relations(release_id,relation,previous_segment_id,segment_id,detail) VALUES(?,?,?,?,?)",
                    (
                        release_id,
                        relation["relation"],
                        relation.get("previous_segment_id"),
                        relation.get("segment_id"),
                        stable_json(relation),
                    ),
                )

    def save_attempt(
        self,
        attempt_id: str,
        training_cutoff: str,
        canonical_fingerprint: str,
        reason: str,
        diagnostics: Mapping[str, Any],
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO segment_attempts VALUES(?,?,?,?,?,?)",
                (
                    attempt_id,
                    training_cutoff,
                    canonical_fingerprint,
                    reason,
                    stable_json(diagnostics),
                    utc_now(),
                ),
            )

    def find_by_fingerprint(self, canonical_fingerprint: str) -> dict[str, Any] | None:
        """幂等：同一份训练数据（与输入顺序无关）命中已发布版本时直接复用。"""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT release_id, bundle FROM segment_releases WHERE canonical_fingerprint=? AND status='published' ORDER BY created_at DESC LIMIT 1",
                (canonical_fingerprint,),
            ).fetchone()
        if row is None:
            return None
        import json

        return {"release_id": row["release_id"], "bundle": json.loads(row["bundle"])}

    def load_release(self, release_id: str) -> dict[str, Any]:
        import json

        with self.connect() as connection:
            row = connection.execute(
                "SELECT bundle FROM segment_releases WHERE release_id=?",
                (release_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"未知发布编号: {release_id}")
            return json.loads(row["bundle"])

    def latest_published(self) -> dict[str, Any] | None:
        import json

        with self.connect() as connection:
            row = connection.execute(
                "SELECT release_id, bundle FROM segment_releases WHERE status='published' ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        return {"release_id": row["release_id"], "bundle": json.loads(row["bundle"])}

    def list_releases(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT release_id,status,training_cutoff,previous_release_id,created_at FROM segment_releases ORDER BY created_at"
            ).fetchall()
        return [dict(row) for row in rows]
