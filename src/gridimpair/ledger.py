from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ClaimResult:
    claimed: bool
    previous_status: str | None = None


class StageLedger:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS stage_runs (
                    stage TEXT NOT NULL,
                    item_id TEXT NOT NULL,
                    config_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    error_type TEXT,
                    artifact_hash TEXT,
                    PRIMARY KEY (stage, item_id, config_hash)
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def claim(
        self, stage: str, item_id: str, config_hash: str, *, resume: bool
    ) -> ClaimResult:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status FROM stage_runs WHERE stage=? AND item_id=? AND config_hash=?",
                (stage, item_id, config_hash),
            ).fetchone()
            previous = row[0] if row else None
            if previous == "completed" and resume:
                connection.commit()
                return ClaimResult(False, previous)
            connection.execute(
                """
                INSERT INTO stage_runs
                    (stage, item_id, config_hash, status, started_at, finished_at, error_type, artifact_hash)
                VALUES (?, ?, ?, 'running', ?, NULL, NULL, NULL)
                ON CONFLICT(stage, item_id, config_hash) DO UPDATE SET
                    status='running', started_at=excluded.started_at,
                    finished_at=NULL, error_type=NULL, artifact_hash=NULL
                """,
                (stage, item_id, config_hash, _now()),
            )
            connection.commit()
            return ClaimResult(True, previous)

    def complete(
        self,
        stage: str,
        item_id: str,
        config_hash: str,
        *,
        artifact_hash: str,
    ) -> None:
        self._finish(stage, item_id, config_hash, "completed", None, artifact_hash)

    def fail(
        self, stage: str, item_id: str, config_hash: str, error: BaseException
    ) -> None:
        self._finish(stage, item_id, config_hash, "failed", type(error).__name__, None)

    def _finish(
        self,
        stage: str,
        item_id: str,
        config_hash: str,
        status: str,
        error_type: str | None,
        artifact_hash: str | None,
    ) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE stage_runs SET status=?, finished_at=?, error_type=?, artifact_hash=?
                WHERE stage=? AND item_id=? AND config_hash=? AND status='running'
                """,
                (
                    status,
                    _now(),
                    error_type,
                    artifact_hash,
                    stage,
                    item_id,
                    config_hash,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("stage item must be claimed before it is finished")
