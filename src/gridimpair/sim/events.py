from __future__ import annotations

import heapq
import random
from dataclasses import replace
from typing import Iterable, Protocol, Sequence

from gridimpair.contracts import MessageEvent, make_message_event


class Impairment(Protocol):
    def apply(self, event: MessageEvent) -> MessageEvent: ...


class EventQueue:
    def __init__(self) -> None:
        self._heap: list[tuple[int, int, MessageEvent]] = []
        self.ledger: list[MessageEvent] = []

    def push(self, event: MessageEvent) -> None:
        self.ledger.append(event)
        if event.dropped:
            return
        if event.actual_delivery_time is None:
            raise ValueError("non-dropped event requires an actual delivery time")
        heapq.heappush(
            self._heap, (event.actual_delivery_time, event.sequence, event)
        )

    def pop_due(self, logical_time: int) -> list[MessageEvent]:
        due: list[MessageEvent] = []
        while self._heap and self._heap[0][0] <= logical_time:
            due.append(heapq.heappop(self._heap)[2])
        return due


class EndpointInbox:
    """Keep the newest causally delivered message for each endpoint."""

    def __init__(self) -> None:
        self._latest: dict[str, MessageEvent] = {}
        self.rejected_stale = 0

    def deliver(self, event: MessageEvent, *, logical_time: int) -> bool:
        if event.dropped:
            raise ValueError("a dropped event cannot be delivered")
        if event.actual_delivery_time is None or event.actual_delivery_time > logical_time:
            raise ValueError("event is not due at the requested logical time")
        prior = self._latest.get(event.endpoint_id)
        if prior is not None and event.sequence <= prior.sequence:
            self.rejected_stale += 1
            return False
        self._latest[event.endpoint_id] = event
        return True

    def payload(self, endpoint_id: str) -> tuple[float, ...] | None:
        event = self._latest.get(endpoint_id)
        return None if event is None else event.payload

    def age_of_information(self, endpoint_id: str, *, logical_time: int) -> int | None:
        event = self._latest.get(endpoint_id)
        if event is None:
            return None
        return logical_time - event.production_time


class BernoulliImpairment:
    def __init__(self, probability: float, *, seed: int) -> None:
        if not 0 <= probability <= 1:
            raise ValueError("probability must be in [0, 1]")
        self.probability = probability
        self._random = random.Random(seed)

    def apply(self, event: MessageEvent) -> MessageEvent:
        if event.dropped or self._random.random() >= self.probability:
            return event
        return replace(event, actual_delivery_time=None, dropped=True)


class BurstLossImpairment:
    """Two-state Gilbert-style loss process.

    ``enter_probability`` enters the bad state and ``exit_probability`` leaves it.
    Every event observed while bad is dropped.
    """

    def __init__(
        self, enter_probability: float, exit_probability: float, *, seed: int
    ) -> None:
        if not 0 <= enter_probability <= 1 or not 0 <= exit_probability <= 1:
            raise ValueError("transition probabilities must be in [0, 1]")
        self.enter_probability = enter_probability
        self.exit_probability = exit_probability
        self._bad = False
        self._random = random.Random(seed)

    def apply(self, event: MessageEvent) -> MessageEvent:
        if self._bad:
            self._bad = self._random.random() >= self.exit_probability
        else:
            self._bad = self._random.random() < self.enter_probability
        if self._bad and not event.dropped:
            return replace(event, actual_delivery_time=None, dropped=True)
        return event


class JitterImpairment:
    def __init__(self, minimum_steps: int, maximum_steps: int, *, seed: int) -> None:
        if minimum_steps > maximum_steps:
            raise ValueError("minimum jitter cannot exceed maximum jitter")
        self.minimum_steps = minimum_steps
        self.maximum_steps = maximum_steps
        self._random = random.Random(seed)

    def apply(self, event: MessageEvent) -> MessageEvent:
        if event.dropped:
            return event
        jitter = self._random.randint(self.minimum_steps, self.maximum_steps)
        delivery = max(
            event.production_time, int(event.actual_delivery_time or 0) + jitter
        )
        return replace(event, actual_delivery_time=delivery)


def generate_event_ledger(
    specs: Iterable[tuple[int, str, tuple[float, ...]]],
    impairments: Sequence[Impairment],
    *,
    fixed_delay: int = 0,
    affected_endpoints: set[str] | None = None,
    active_intervals: Sequence[tuple[int, int]] | None = None,
) -> list[MessageEvent]:
    if fixed_delay < 0:
        raise ValueError("fixed_delay must be non-negative")
    ledger: list[MessageEvent] = []
    for sequence, (production_time, endpoint_id, payload) in enumerate(specs):
        intended = production_time
        endpoint_is_affected = (
            affected_endpoints is None or endpoint_id in affected_endpoints
        )
        interval_is_active = (
            active_intervals is None
            or any(start <= production_time < stop for start, stop in active_intervals)
        )
        affected = endpoint_is_affected and interval_is_active
        event = make_message_event(
            production_time=production_time,
            intended_delivery_time=intended,
            actual_delivery_time=production_time + (fixed_delay if affected else 0),
            sequence=sequence,
            endpoint_id=endpoint_id,
            payload=payload,
            dropped=False,
        )
        if affected:
            for impairment in impairments:
                event = impairment.apply(event)
        ledger.append(event)
    return ledger
