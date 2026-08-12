from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from gridimpair.contracts import MessageEvent, ScenarioConfig, make_message_event, stable_hash
from gridimpair.ledger import StageLedger


def test_stable_hash_ignores_mapping_insertion_order() -> None:
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})


def test_message_factory_hashes_payload_canonically() -> None:
    event = make_message_event(
        production_time=1,
        intended_delivery_time=2,
        actual_delivery_time=3,
        sequence=4,
        endpoint_id="der-1",
        payload=(0.25, -0.0),
        dropped=False,
    )
    assert len(event.payload_hash) == 64
    clone = MessageEvent(**asdict(event))
    assert clone == event


def test_scenario_rejects_invalid_probability() -> None:
    try:
        ScenarioConfig(
            scenario_id="s",
            pair_id="p",
            split="development",
            regime="smooth",
            delay_steps=0,
            drop_probability=1.2,
            endpoint_fraction=0.5,
            status="impaired",
            profile_seed=1,
            schedule_seed=2,
            impairment_seed=3,
        )
    except ValueError as exc:
        assert "drop_probability" in str(exc)
    else:
        raise AssertionError("invalid probability was accepted")


def test_ledger_completed_item_is_skipped_only_for_same_hash(tmp_path: Path) -> None:
    ledger = StageLedger(tmp_path / "ledger.sqlite")
    assert ledger.claim("audit", "source", "hash-a", resume=True).claimed
    ledger.complete("audit", "source", "hash-a", artifact_hash="out")
    assert not ledger.claim("audit", "source", "hash-a", resume=True).claimed
    assert ledger.claim("audit", "source", "hash-b", resume=True).claimed


def test_ledger_failed_item_can_be_reclaimed(tmp_path: Path) -> None:
    ledger = StageLedger(tmp_path / "ledger.sqlite")
    ledger.claim("simulate", "s1", "cfg", resume=True)
    ledger.fail("simulate", "s1", "cfg", RuntimeError("boom"))
    result = ledger.claim("simulate", "s1", "cfg", resume=True)
    assert result.claimed
    assert result.previous_status == "failed"
