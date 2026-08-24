from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from plastic_heredity.phir_extension_px3 import (
    ARMS,
    CONFIRMATION_BRANCHES,
    CONFIRMATION_CPU_SECONDS,
    CONFIRMATION_MAX_WORKERS,
    CONFIRMATION_MATRICES,
    DEVELOPMENT_BRANCHES,
    DEVELOPMENT_CPU_SECONDS,
    DEVELOPMENT_CARRIED_FORWARD,
    DEVELOPMENT_EDITS,
    DEVELOPMENT_MAX_WORKERS,
    DEVELOPMENT_MATRICES,
    HALVES,
    RIDGE_GRID,
    DevelopmentBatch,
    FrozenPhiSurrogate,
    _batch_digest,
    _confirmation_future_seed,
    _confirmation_selection_seed,
    confirmation_spec,
    development_spec,
    fit_surrogates,
    load_surrogates,
    program_protocol,
    save_surrogates,
    smoke_spec,
    validation_checks,
)


def test_px3_design_is_fixed_and_bounded() -> None:
    development = development_spec()
    confirmation = confirmation_spec()
    assert DEVELOPMENT_MATRICES == 12 and development.matrices == 12
    assert DEVELOPMENT_EDITS == 24 and DEVELOPMENT_BRANCHES == 16
    assert DEVELOPMENT_CPU_SECONDS == 104 * 3600
    assert DEVELOPMENT_MAX_WORKERS == 8
    assert DEVELOPMENT_CARRIED_FORWARD == tuple(range(6))
    assert CONFIRMATION_MATRICES == 24 and confirmation.matrices == 24
    assert CONFIRMATION_BRANCHES == 64
    assert CONFIRMATION_CPU_SECONDS == 64 * 3600
    assert CONFIRMATION_MAX_WORKERS == 8
    assert HALVES == {"A": (0, 32), "B": (32, 64)}
    assert ARMS == ("PHI_UP", "PHI_DOWN", "RANDOM", "NOOP")
    assert RIDGE_GRID == (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)


def test_confirmation_future_stream_is_arm_free_and_selection_separate() -> None:
    assert "arm" not in inspect.signature(_confirmation_future_seed).parameters
    spec = smoke_spec("smoke-test")
    future = _confirmation_future_seed(spec, "02", 4, 2, 3)
    assert len({future for _arm in ARMS}) == 1
    assert future != _confirmation_selection_seed(spec, "02", 4, 2)


def test_batch_digest_excludes_cpu_but_includes_science() -> None:
    provisional = DevelopmentBatch(1, ({"target": 2.0},), ({"state": "a"},), 1.0, "")
    first = DevelopmentBatch(
        provisional.matrix_id,
        provisional.training_rows,
        provisional.state_rows,
        provisional.cpu_seconds,
        _batch_digest(provisional),
    )
    changed_time = DevelopmentBatch(
        first.matrix_id,
        first.training_rows,
        first.state_rows,
        999.0,
        first.scientific_digest,
    )
    changed_science = DevelopmentBatch(1, ({"target": 3.0},), first.state_rows, 1.0, "")
    assert _batch_digest(first) == _batch_digest(changed_time)
    assert _batch_digest(first) != _batch_digest(changed_science)


def _synthetic_training() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    edit_values = np.asarray((-3.0, -2.0, -1.0, 1.0, 2.0, 3.0))
    for candidate in ("02", "03"):
        candidate_scale = 1.0 if candidate == "02" else 1.5
        for matrix_id in range(10):
            for landmark in (20, 40):
                for edit_index, value in enumerate(edit_values):
                    feature = np.zeros(195, dtype=np.float64)
                    feature[0] = value
                    feature[1] = value * 0.25
                    rows.append(
                        {
                            "matrix_id": matrix_id,
                            "candidate": candidate,
                            "replicate": 0,
                            "landmark": landmark,
                            "state_id": f"{candidate}-{matrix_id}-{landmark}",
                            "edit_index": edit_index,
                            "remove_type": edit_index,
                            "add_type": edit_index + 1,
                            "feature": feature.tolist(),
                            "realized_phi": candidate_scale * value,
                            "noop_phi": 0.0,
                            "target_delta_phi": candidate_scale * value,
                        }
                    )
    return pd.DataFrame(rows)


def test_ridge_development_is_matrix_cross_fitted_and_serializable(tmp_path) -> None:
    models, diagnostics, scored = fit_surrogates(_synthetic_training())
    assert diagnostics["development_gate"]
    assert set(models) == {"02", "03"}
    assert np.isfinite(scored["oof_predicted_delta_phi"]).all()
    archive = tmp_path / "models.npz"
    contract = tmp_path / "contract.json"
    save_surrogates(models, archive, contract)
    restored = load_surrogates(archive, contract)
    probe = np.zeros((2, 195), dtype=np.float64)
    probe[:, 0] = (-1.0, 1.0)
    for candidate in models:
        assert np.array_equal(models[candidate].coefficient, restored[candidate].coefficient)
        assert np.array_equal(models[candidate].predict(probe), restored[candidate].predict(probe))


def test_program_keeps_failed_development_and_claim_boundaries() -> None:
    protocol = program_protocol()
    assert protocol["development"]["pca"] is False
    assert protocol["confirmation"]["all_legal_edits_scored"]
    assert protocol["run_confirmation_if_development_fails"]
    assert protocol["failed_development_cannot_be_rescued"]
    assert protocol["no_48_matrix_campaign"]
    assert "strict-eight is excluded" in protocol["claim_boundary"]
    assert protocol["development"]["interim_diagnostics_inspected"]
    assert protocol["development"]["original_development_gate_eligible"] is False
    assert protocol["classification"]["original_px3_confirmed"] is False


def test_complete_px3_validation_suite() -> None:
    checks = validation_checks()
    assert len(checks) >= 18
    assert all(checks.values()), [name for name, passed in checks.items() if not passed]
