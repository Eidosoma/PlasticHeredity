from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from plastic_heredity import phir_extension_px9 as px9
from plastic_heredity import phir_extension_px11 as px11
from plastic_heredity.config import GardConfig
from plastic_heredity.intervention_core import MolecularEdit
from plastic_heredity.phir_instruments import ATOM_NAMES
from plastic_heredity.simulator import Snapshot, generate_beta


def _case() -> px9.ResilienceCase:
    config = GardConfig()
    rng = np.random.default_rng(111)
    beta = generate_beta(config, rng)
    composition = np.zeros(config.n_types, dtype=np.int64)
    composition[:40] = 1
    return px9.ResilienceCase(
        "px11-test",
        "02",
        0,
        20,
        beta,
        Snapshot(composition, 20, (False,), (0.8,)),
        np.tile(composition, (32, 1)).astype(np.int16),
    )


def test_protocol_freezes_staged_manual_design() -> None:
    value = px11.protocol()
    assert px11.PILOT_MATRICES == 24
    assert px11.CONFIRMATION_MATRICES == 48
    assert px11.BRANCHES == 128
    assert value["cohorts"]["manual_stop_after_pilot"] is True
    assert value["cohorts"]["automatic_confirmation"] is False
    assert value["public_phi_r_can_win"] is False


def test_arm_panel_and_three_registered_contrasts_are_exact() -> None:
    assert px11.ARMS == (
        "Q00",
        "Q20",
        "Q40",
        "Q60",
        "Q80",
        "Q100",
        "RANDOM_SWAP",
        "RULE_STABILIZE",
        "RULE_DESTABILIZE",
        "TIGHTEN",
        "LOOSEN",
        "BLOCK_RANDOM",
        "NOOP",
    )
    assert px11.CONTRASTS == {
        "model": ("Q100", "Q00"),
        "physical_rule": ("RULE_STABILIZE", "RULE_DESTABILIZE"),
        "beta_surgery": ("TIGHTEN", "LOOSEN"),
    }


def test_future_seed_contract_is_arm_free_and_context_restores() -> None:
    assert "arm" not in inspect.signature(px9._future_seed).parameters
    old_label = px9.LABEL
    old_domains = px9.SEED_DOMAINS
    old_matrices = px9.MATRICES
    with px11._px9_seed_context(px11.pilot_spec()):
        assert px9.LABEL == px11.LABEL
        assert px9.SEED_DOMAINS == px11.SEED_DOMAINS
        assert px9.MATRICES == 24
    assert px9.LABEL == old_label
    assert px9.SEED_DOMAINS is old_domains
    assert px9.MATRICES == old_matrices


def test_confirmation_active_arm_reduction_is_deterministic() -> None:
    contract = {
        "selected_sensor_profile": "COMPACT",
        "advancing_families": ["physical_rule"],
        "dose_channel_active": False,
        "sensor_active": False,
    }
    spec = px11.confirmation_spec(contract)
    assert px11._active_arms(spec) == (
        "RANDOM_SWAP",
        "RULE_STABILIZE",
        "RULE_DESTABILIZE",
        "NOOP",
    )


def test_iri_is_symmetric_under_part_label_exchange() -> None:
    atoms = {f"atom_{name}": 0.0 for name in ATOM_NAMES}
    atoms.update(
        {
            "atom_s_to_u0": 0.2,
            "atom_s_to_u1": 0.4,
            "atom_u0_to_s": 0.3,
            "atom_u1_to_s": 0.1,
            "atom_r_to_r": -0.2,
            "atom_s_to_s": -0.3,
            "atom_u0_to_u1": -0.1,
            "atom_u1_to_u0": -0.05,
        }
    )

    def iri(values: dict[str, float]) -> float:
        groups = {
            name: sum(values[column] for column in columns)
            for name, columns in px11.GROUP_COLUMNS.items()
        }
        return (
            groups["downward_routing"]
            + groups["upward_integration"]
            - groups["redundant_persistence"]
            - groups["synergy_persistence"]
            - groups["cross_part_transfer"]
        )

    swapped = dict(atoms)
    for left, right in (
        ("atom_s_to_u0", "atom_s_to_u1"),
        ("atom_u0_to_s", "atom_u1_to_s"),
        ("atom_u0_to_u1", "atom_u1_to_u0"),
    ):
        swapped[left], swapped[right] = swapped[right], swapped[left]
    assert iri(atoms) == iri(swapped)


def test_observation_masks_are_deterministic_and_outcome_independent() -> None:
    case = _case()
    spec = px11.smoke_spec()
    profile = px11.SENSOR_BY_NAME["COORD25"]
    first = px11._coordinate_mask(case, profile, spec)
    second = px11._coordinate_mask(case, profile, spec)
    assert np.array_equal(first, second)
    assert first.dtype == bool
    assert first.sum() == 25


def test_count_thinning_is_replayed_exactly_and_preserves_integers() -> None:
    case = _case()
    spec = px11.smoke_spec()
    values = np.tile(case.snapshot.composition, (4, 1))
    mask = np.ones(100, dtype=bool)
    profile = px11.SENSOR_BY_NAME["COUNT50"]
    first = px11._observe_rows(values, mask, profile, spec, case, 3, "past")
    second = px11._observe_rows(values, mask, profile, spec, case, 3, "past")
    assert np.array_equal(first, second)
    assert np.issubdtype(first.dtype, np.integer)
    assert np.all(first >= 0)
    assert np.all(first <= values)


def test_shared_dose_fit_recovers_positive_synthetic_gradient() -> None:
    rng = np.random.default_rng(112)
    rows: list[dict[str, object]] = []
    for state in range(20):
        for index, arm in enumerate((*px11.QUANTILE_ARMS, "NOOP")):
            shift = float(index if arm != "NOOP" else 2.5)
            probability = 0.08 + 0.10 * index if arm != "NOOP" else 0.30
            for branch in range(64):
                rows.append(
                    {
                        "state_id": f"s{state}",
                        "matrix_id": state // 2,
                        "arm": arm,
                        "primary": int(rng.random() < probability),
                        "predicted_shift": shift,
                    }
                )
    fitted = px11._fit_dose_coefficient(pd.DataFrame(rows))
    assert fitted["converged"] is True
    assert fitted["coefficient"] > 0.0
    assert fitted["shift_scale"] > 0.0


def test_bootstrap_keeps_matrix_as_the_only_resampling_unit() -> None:
    spec = px11.smoke_spec()
    arrays: dict[str, np.ndarray] = {}
    values = pd.Series([0.1, 0.2, 0.3], index=[2, 5, 9])
    result = px11._bootstrap_summary(values, spec, "fixture", arrays)
    assert result["matrices"] == 3
    assert np.array_equal(arrays["fixture__matrix_ids"], [2, 5, 9])
    assert arrays["fixture__bootstrap"].shape == (spec.bootstrap_draws,)


def test_public_phi_r_is_not_part_of_advancement_contract() -> None:
    text = px11.protocol()["information_redistribution"]
    assert text["iri_is_phi_r"] is False
    assert text["public_nine_atom_negative_control"] is True


def test_retained_cr6_trace_reproduces_one_archived_branch() -> None:
    regime = "POS_A_M4_S5"
    directory = px11.CR6_ROOT / regime
    with np.load(directory / "state_and_matrix_arrays.npz", allow_pickle=False) as z:
        arrays = {name: z[name] for name in z.files}
    beta = np.asarray(arrays["beta"][0], dtype=float)
    case = px11._cr6_state_case(arrays, 0, beta)
    selection = pd.read_csv(directory / "selected_interventions.csv")
    selected = selection[
        (selection["state_id"] == case.state_id)
        & (selection["arm"] == "MODEL_UP")
    ].iloc[0]
    edit = MolecularEdit(int(selected["remove_type"]), int(selected["add_type"]))
    _block, digest = px11._trace_cr6_branch(
        case, regime, "MODEL_UP", edit, 0
    )
    archived = pd.read_csv(directory / "branches.csv.gz", nrows=1).iloc[0]
    assert digest == archived["record_digest"]
