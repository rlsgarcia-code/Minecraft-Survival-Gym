"""Rebuildable SQLite index over immutable episode manifests."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable

from .contracts import EpisodeManifestV2, QualityReportV1
from .storage import ResearchStorage, verify_episode_artifacts


class DatasetCatalog:
    def __init__(self, storage: ResearchStorage, database: Path) -> None:
        self.storage = storage
        self.database = database
        database.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(database, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                environment_id TEXT NOT NULL,
                world_id TEXT NOT NULL,
                world_snapshot_id TEXT,
                state TEXT NOT NULL,
                data_tiers_json TEXT NOT NULL,
                actor_ids_json TEXT NOT NULL,
                step_count INTEGER NOT NULL,
                frame_count INTEGER NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                path TEXT NOT NULL UNIQUE,
                manifest_json TEXT NOT NULL,
                quality_json TEXT
            );
            CREATE INDEX IF NOT EXISTS episodes_session ON episodes(session_id);
            CREATE INDEX IF NOT EXISTS episodes_world ON episodes(world_id, world_snapshot_id);
            """
        )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def rebuild(self) -> dict[str, int]:
        entries = list(self._manifest_paths())
        accepted = quarantined = invalid = 0
        with self._lock:
            with self._connection:
                self._connection.execute("DELETE FROM episodes")
                for path in entries:
                    try:
                        manifest = EpisodeManifestV2.model_validate_json(
                            path.read_text(encoding="utf-8")
                        )
                        quality_path = path.parent / str(manifest.quality_report_ref or "")
                        quality = (
                            QualityReportV1.model_validate_json(
                                quality_path.read_text(encoding="utf-8")
                            )
                            if quality_path.is_file()
                            else None
                        )
                        if verify_episode_artifacts(path.parent, manifest):
                            invalid += 1
                            continue
                    except (OSError, ValueError):
                        invalid += 1
                        continue
                    self._upsert(manifest, quality, path.parent)
                    if manifest.state == "accepted":
                        accepted += 1
                    else:
                        quarantined += 1
        return {
            "indexed": accepted + quarantined,
            "accepted": accepted,
            "quarantined": quarantined,
            "invalid": invalid,
        }

    def list_episodes(
        self,
        *,
        state: str | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM episodes"
        parameters: tuple[Any, ...] = ()
        if state:
            sql += " WHERE state = ?"
            parameters = (state,)
        sql += " ORDER BY started_at DESC LIMIT ?"
        with self._lock:
            rows = self._connection.execute(
                sql, (*parameters, max(1, min(limit, 5000)))
            ).fetchall()
            return [self._project(row) for row in rows]

    def get_episode(self, episode_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)
            ).fetchone()
        return self._project(row) if row else None

    def accepted_manifests(self, episode_ids: Iterable[str]) -> list[EpisodeManifestV2]:
        results: list[EpisodeManifestV2] = []
        for episode_id in episode_ids:
            item = self.get_episode(episode_id)
            if item is None:
                raise KeyError(episode_id)
            if item["state"] != "accepted":
                raise ValueError(f"episode is not accepted: {episode_id}")
            results.append(EpisodeManifestV2.model_validate(item["manifest"]))
        return results

    def _manifest_paths(self) -> Iterable[Path]:
        roots = [
            self.storage.spool_root / "accepted",
            self.storage.spool_root / "quarantine",
        ]
        if self.storage.external_mounted:
            roots.extend(
                [
                    self.storage.external_root / "raw",
                    self.storage.external_root / "quarantine",
                ]
            )
        for root in roots:
            if root.is_dir():
                yield from root.glob("*/manifest.json")

    def _upsert(
        self,
        manifest: EpisodeManifestV2,
        quality: QualityReportV1 | None,
        root: Path,
    ) -> None:
        self._connection.execute(
            """
            INSERT OR REPLACE INTO episodes (
                episode_id, session_id, environment_id, world_id,
                world_snapshot_id, state, data_tiers_json, actor_ids_json,
                step_count, frame_count, started_at, ended_at, path,
                manifest_json, quality_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                manifest.episode_id,
                manifest.session_id,
                manifest.environment_id,
                manifest.world_id,
                manifest.world_snapshot_id,
                manifest.state,
                json.dumps(manifest.data_tiers),
                json.dumps([item.actor_id for item in manifest.actors]),
                manifest.step_count,
                manifest.frame_count,
                manifest.started_at.isoformat(),
                manifest.ended_at.isoformat() if manifest.ended_at else None,
                str(root),
                manifest.model_dump_json(),
                quality.model_dump_json() if quality else None,
            ),
        )

    @staticmethod
    def _project(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "episode_id": row["episode_id"],
            "session_id": row["session_id"],
            "environment_id": row["environment_id"],
            "world_id": row["world_id"],
            "world_snapshot_id": row["world_snapshot_id"],
            "state": row["state"],
            "data_tiers": json.loads(row["data_tiers_json"]),
            "actor_ids": json.loads(row["actor_ids_json"]),
            "step_count": row["step_count"],
            "frame_count": row["frame_count"],
            "started_at": row["started_at"],
            "ended_at": row["ended_at"],
            "path": row["path"],
            "manifest": json.loads(row["manifest_json"]),
            "quality": json.loads(row["quality_json"]) if row["quality_json"] else None,
        }
