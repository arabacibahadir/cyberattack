import numpy as np

from gridimpair.r1 import metadata as md


def _event(t, ep, seq, delay=0, dropped=False):
    return {"production_time": t, "actual_delivery_time": None if dropped else t + delay,
            "sequence": seq, "endpoint_id": ep, "dropped": dropped}


def _ledger(steps=6):
    events, seq = [], 0
    for t in range(steps):
        for ep in ("a", "b"):
            events.append(_event(t, ep, seq, delay=2 if (ep == "b" and t == 2) else 0, dropped=(ep == "a" and t == 3)))
            seq += 1
        events.append(_event(t + 1, md.TELEMETRY, 100 + t))
    return events


def test_replay_matches_hand_computed_bookkeeping():
    out = md.replay(_ledger(), 6)
    ages_b_t2 = 2 - 1  # b's t=2 message arrives at t=4, so at t=2 latest is t=1
    assert out[1, 2] == ages_b_t2
    assert out[1, 3] == 1  # a dropped at t=3 (latest a = t2 -> age 1); b on time -> age 0
    assert out[2, 0] == 1.0
    assert out[4, 0] == 1.0  # no telemetry yet -> t + 1
    assert out[4, 1] == 0.0


def test_late_message_with_older_sequence_counts_as_stale():
    ledger = _ledger()
    out = md.replay(ledger, 6)
    # b's t=2 message (seq 5) arrives at t=4 after b's t=3 message (seq 7) -> rejected
    assert out[3, -1] == 1


def test_timestamp_dropout_keeps_previous_age():
    full = md.replay(_ledger(), 6, md.Perturbation("timestamp_dropout", 1.0), np.random.default_rng(0))
    assert np.all(full[4] == np.arange(6) + 1)  # never a usable telemetry timestamp


def test_metadata_delay_shifts_cyber_channels():
    columns = list(md.CYBER) + ["phys"]
    raw = np.vstack([np.tile(np.arange(10.0), (5, 1)), np.zeros(10)])
    out = md.perturb_trace(raw, columns, [], md.Perturbation("metadata_delay", 3), np.random.default_rng(0), ["phys"], np.ones(1))
    assert out[0, 5] == 2.0 and out[0, 0] == 0.0
    assert np.all(out[-1] == 0.0)


def test_physical_noise_leaves_cyber_untouched():
    columns = list(md.CYBER) + ["phys"]
    raw = np.vstack([np.ones((5, 50)), np.zeros(50)])
    out = md.perturb_trace(raw, columns, [], md.Perturbation("physical_noise", 1.0), np.random.default_rng(1), ["phys"], np.ones(1))
    assert np.all(out[:5] == 1.0) and out[-1].std() > 0.5
