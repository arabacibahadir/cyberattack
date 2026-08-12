from __future__ import annotations

import pandas as pd

from gridimpair.workflow import _training_signal_gate


def test_training_gate_uses_preregistered_candidate_lift_on_validation() -> None:
    frame = pd.DataFrame(
        {
            "model": ["candidate"] * 4,
            "split": ["validation_a"] * 4,
            "label": [0, 0, 1, 1],
            "raw_score": [0.1, 0.2, 0.8, 0.9],
        }
    )

    result = _training_signal_gate(frame, minimum_auprc_lift=0.02)

    assert result["decision"] == "GO"
    assert result["selection_split"] == "validation_a"
