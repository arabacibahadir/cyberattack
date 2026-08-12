from __future__ import annotations

import json
from pathlib import Path

import pytest

from gridimpair.contracts import stable_hash
from gridimpair.evaluation.closed_loop import LockedManifestReader, hash_unlock_token


def test_locked_reader_refuses_access_before_valid_single_use_unlock(
    tmp_path: Path,
) -> None:
    manifest = {"pairs": [{"pair_id": "p1", "onset": 4, "offset": 8}]}
    manifest_path = tmp_path / "locked.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    token = "one-time-secret"
    guard_path = tmp_path / "unlock.json"
    guard_path.write_text(
        json.dumps(
            {
                "token_hash": hash_unlock_token(token),
                "config_hash": "cfg",
                "split_hash": "split",
                "manifest_hash": stable_hash(manifest),
            }
        ),
        encoding="utf-8",
    )
    reader = LockedManifestReader(manifest_path, guard_path)
    with pytest.raises(PermissionError, match="unlock"):
        reader.read()
    with pytest.raises(PermissionError, match="hash"):
        reader.unlock(token, config_hash="wrong", split_hash="split")
    reader.unlock(token, config_hash="cfg", split_hash="split")
    assert reader.read() == manifest

    second_reader = LockedManifestReader(manifest_path, guard_path)
    with pytest.raises(PermissionError, match="consumed"):
        second_reader.unlock(token, config_hash="cfg", split_hash="split")


def test_closed_loop_config_rejects_test_time_adaptation() -> None:
    from gridimpair.evaluation.closed_loop import ClosedLoopConfig

    with pytest.raises(ValueError, match="adaptation"):
        ClosedLoopConfig(on_policy_retraining=False, test_time_adaptation=True)
