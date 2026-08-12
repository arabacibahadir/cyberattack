from __future__ import annotations

from gridimpair.contracts import make_message_event
from gridimpair.sim.events import (
    BernoulliImpairment,
    BurstLossImpairment,
    EndpointInbox,
    EventQueue,
    JitterImpairment,
    generate_event_ledger,
)


def _event(sequence: int, delivery: int | None, *, dropped: bool = False):
    return make_message_event(
        production_time=0,
        intended_delivery_time=0,
        actual_delivery_time=delivery,
        sequence=sequence,
        endpoint_id="e1",
        payload=(float(sequence),),
        dropped=dropped,
    )


def test_future_message_is_not_visible() -> None:
    queue = EventQueue()
    queue.push(_event(7, 3))
    assert queue.pop_due(2) == []
    assert [item.sequence for item in queue.pop_due(3)] == [7]


def test_delivery_ties_are_broken_by_sequence() -> None:
    queue = EventQueue()
    queue.push(_event(2, 1))
    queue.push(_event(1, 1))
    assert [item.sequence for item in queue.pop_due(1)] == [1, 2]


def test_dropped_message_is_logged_but_not_queued() -> None:
    queue = EventQueue()
    queue.push(_event(1, None, dropped=True))
    assert len(queue.ledger) == 1
    assert queue.pop_due(100) == []


def test_seeded_impairments_replay_identically() -> None:
    impairments_a = [
        BernoulliImpairment(0.2, seed=7),
        BurstLossImpairment(0.1, 0.4, seed=8),
        JitterImpairment(-1, 2, seed=9),
    ]
    impairments_b = [
        BernoulliImpairment(0.2, seed=7),
        BurstLossImpairment(0.1, 0.4, seed=8),
        JitterImpairment(-1, 2, seed=9),
    ]
    specs = [(time, "e1", (float(time),)) for time in range(20)]
    assert generate_event_ledger(specs, impairments_a) == generate_event_ledger(
        specs, impairments_b
    )


def test_impairment_can_be_restricted_to_selected_endpoints() -> None:
    specs = [(0, "e1", (1.0,)), (0, "e2", (2.0,))]
    ledger = generate_event_ledger(
        specs,
        [BernoulliImpairment(1.0, seed=1)],
        fixed_delay=2,
        affected_endpoints={"e1"},
    )
    assert ledger[0].dropped
    assert not ledger[1].dropped
    assert ledger[1].actual_delivery_time == 0


def test_age_of_information_uses_message_production_time() -> None:
    inbox = EndpointInbox()
    delayed = make_message_event(
        production_time=2,
        intended_delivery_time=2,
        actual_delivery_time=5,
        sequence=7,
        endpoint_id="e1",
        payload=(11.0,),
        dropped=False,
    )
    assert inbox.deliver(delayed, logical_time=5)
    assert inbox.age_of_information("e1", logical_time=5) == 3


def test_late_stale_message_cannot_overwrite_newer_payload() -> None:
    inbox = EndpointInbox()
    newer = make_message_event(
        production_time=4,
        intended_delivery_time=4,
        actual_delivery_time=4,
        sequence=9,
        endpoint_id="e1",
        payload=(90.0,),
        dropped=False,
    )
    stale = make_message_event(
        production_time=3,
        intended_delivery_time=3,
        actual_delivery_time=5,
        sequence=8,
        endpoint_id="e1",
        payload=(80.0,),
        dropped=False,
    )
    assert inbox.deliver(newer, logical_time=4)
    assert not inbox.deliver(stale, logical_time=5)
    assert inbox.payload("e1") == (90.0,)
    assert inbox.rejected_stale == 1
