"""Prospective confirmation of the pilot-developed direct/hurdle ensemble.

This workflow is intentionally narrower than :mod:`regime_prediction`.  The
registered pilot stopped before confirmation because no single model family
met its stability gate.  After that result was known, an equal-probability
ensemble of the already fitted direct and hurdle models was proposed.  This
module can seal that post-pilot proposal and score it once on a new cohort; it
contains no model fitting, recalibration, or model-selection path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits

from .config import CANDIDATES
from .experiment import _json_ready, _runtime_manifest
from .mechanistic import (
    _atomic_destination,
    _canonical_digest,
    sha256_file,
    verify_checksums,
    write_checksums,
)
from .mechanistic_metrics import (
    _paired_gain,
    _state_brier,
    _state_log_loss,
    holm_adjust,
    paired_matrix_randomization_p,
)
from .metrics import centered_spearman, spearman
from .regime_prediction import (
    BOOTSTRAP_REPETITIONS,
    BRANCHES,
    CONFIRMATION_MATRICES,
    HORIZON,
    LANDMARKS,
    MINIMUM_EVENT_MATRICES,
    MINIMUM_EVENTS,
    PILOT_FORMAT,
    PRIMARY_ENDPOINT,
    RANDOMIZATION_REPETITIONS,
    SOURCE_FILES as PREDICTION_SOURCE_FILES,
    PredictionCase,
    _campaign_status,
    _candidate_mask,
    _experiment,
    _labels,
    _matrix_ids,
    _power,
    _protocol as prediction_protocol,
    _save_arrays,
    _seed_domains as prediction_seed_domains,
    _source_hashes as prediction_source_hashes,
    _strict_labels_from_branch_table,
    _write_branch_tables,
    build_prediction_cohort,
    read_checkpoint_status,
    replay_audit,
    run_prediction_branches,
    verify_design,
)
from .regime_prediction_features import (
    PredictionRawFeatures,
    extract_prediction_features,
)
from .regime_prediction_models import (
    HurdleModel,
    MODEL_FAMILIES,
    PredictionFamilyModel,
    SequentialRidgeModel,
    model_summary,
)
from .seeds import derive_seed

FloatArray = NDArray[np.float64]

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REGISTRATION_FORMAT = "plastic-heredity-regime-ensemble-registration-v1"
CONFIRMATION_FORMAT = "plastic-heredity-regime-ensemble-confirmation-v1"
CAMPAIGN_FORMAT = "plastic-heredity-regime-ensemble-campaign-v1"

COHORT_LABEL = "REGENSCONF"
CONFIRMATION_MASTER_SEED = (
    "650fecf3a953484eea76701964a08f85d81f54a0bbe7ff5e0e30ca05fd02537d"
)
BOOTSTRAP_MASTER_SEED = (
    "a2e1b1d3eab6e5f0fe4cc18d3ee8f4800b41a1d53efd53db61fbf05d77a822be"
)
RANDOMIZATION_MASTER_SEED = (
    "31e93c0ce240110a7b91bf3c1aa85f807632211e56bca2c074e0120797107203"
)

SOURCE_FILES = tuple(
    dict.fromkeys(
        (*PREDICTION_SOURCE_FILES, "plastic_heredity/regime_ensemble_confirmation.py")
    )
)


def _source_hashes() -> dict[str, str]:
    return {name: sha256_file(REPOSITORY_ROOT / name) for name in SOURCE_FILES}


def _seed_domains() -> dict[str, str]:
    return {
        **prediction_seed_domains(),
        "ensemble_confirmation": CONFIRMATION_MASTER_SEED,
        "ensemble_bootstrap": BOOTSTRAP_MASTER_SEED,
        "ensemble_randomization": RANDOMIZATION_MASTER_SEED,
    }


def _protocol() -> dict[str, Any]:
    old_endpoint = prediction_protocol()["endpoint_contract"]
    experiment = _experiment(CONFIRMATION_MASTER_SEED, CONFIRMATION_MATRICES)
    value: dict[str, Any] = {
        "format": REGISTRATION_FORMAT,
        "status": "sealed_before_confirmation_matrix_generation",
        "experiment_role": (
            "prospective confirmation of a pilot-developed direct-plus-hurdle "
            "ensemble; the pilot's registered failure remains unchanged"
        ),
        "primary_endpoint": PRIMARY_ENDPOINT,
        "endpoint_contract": old_endpoint,
        "frozen_predictor": {
            "formula": "p_ensemble = 0.5 * p_direct + 0.5 * p_hurdle",
            "scale": "probability",
            "direct_weight": 0.5,
            "hurdle_weight": 0.5,
            "candidate_specific_models": True,
            "same_family_recipe_for_both_candidates": True,
            "comparator": "the identical frozen h10 baseline in both families",
            "source": "already fitted final models from the checksum-sealed pilot",
        },
        "prohibited_after_registration": [
            "model refitting",
            "preprocessing refitting",
            "recalibration",
            "regularization search",
            "ensemble-weight fitting",
            "candidate-specific family switching",
            "nonlinear fallback",
            "endpoint or threshold changes",
        ],
        "cohort": {
            "experiment": experiment.to_dict(),
            "label": COHORT_LABEL,
            "matrices": CONFIRMATION_MATRICES,
            "candidates": list(CANDIDATES),
            "landmarks": list(LANDMARKS),
            "states": CONFIRMATION_MATRICES * len(CANDIDATES) * len(LANDMARKS),
            "branches_per_state": BRANCHES,
            "future_horizon_fissions": HORIZON,
            "primary_futures": (
                CONFIRMATION_MATRICES * len(CANDIDATES) * len(LANDMARKS) * BRANCHES
            ),
            "full_exact_replay": True,
            "new_seed_domain": CONFIRMATION_MASTER_SEED,
            "number_of_confirmation_cohorts": 1,
        },
        "primary_inference": {
            "cells": "candidate 02/03 x fixed branch half A/B",
            "contrast": "frozen ensemble minus frozen h10 comparator",
            "loss": "natural-log Bernoulli log loss on state-level event fractions",
            "cluster_unit": "whole catalytic matrix",
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "randomization_repetitions": RANDOMIZATION_REPETITIONS,
            "multiplicity": "Holm adjustment over the four primary cells",
            "gate_per_cell": [
                "log-loss gain > 0",
                "whole-matrix bootstrap 95% lower bound > 0",
                "Holm-adjusted whole-matrix randomization p < 0.05",
            ],
            "overall_gate": "all three conditions in all four cells",
            "validity_prerequisite": {
                "minimum_events_per_candidate": MINIMUM_EVENTS,
                "minimum_event_matrices_per_candidate": MINIMUM_EVENT_MATRICES,
            },
        },
        "secondary_no_rescue": [
            "direct and hurdle constituent performance",
            "Brier-score contrast",
            "rank descriptives",
            "event and stage incidence",
        ],
        "operations": {
            "registration_verified_before_any_matrix_generation": True,
            "per_state_resumable_checkpoints": True,
            "generation_and_replay_checkpoints_separate": True,
            "round_trip_primary_metric_recomputation": True,
            "refuse_output_overwrite": True,
        },
        "claim_boundary": {
            "tested": (
                "out-of-cohort prediction of strict break-and-distinct-renewal "
                "probability beyond unique history and growth clocks"
            ),
            "not_tested": [
                "causal control",
                "molecular intervention",
                "attractor or regime switching",
                "recurrence",
                "origin-of-life realism",
            ],
        },
    }
    value["protocol_id"] = _canonical_digest(value)
    return value


def _load_pilot_family_archive(
    path: Path,
) -> dict[str, dict[str, PredictionFamilyModel]]:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, dict) or set(value) != set(MODEL_FAMILIES):
        raise ValueError("pilot family archive does not contain the registered menu")
    for family in MODEL_FAMILIES:
        if not isinstance(value[family], dict) or set(value[family]) != set(CANDIDATES):
            raise ValueError(f"invalid pilot models for family {family}")
        for candidate in CANDIDATES:
            model = value[family][candidate]
            if (
                not isinstance(model, PredictionFamilyModel)
                or model.family != family
                or model.candidate != candidate
            ):
                raise ValueError(f"invalid pilot model {family}/{candidate}")
    return value


def _validate_frozen_models(
    value: Any,
) -> dict[str, dict[str, PredictionFamilyModel]]:
    if not isinstance(value, dict) or set(value) != set(CANDIDATES):
        raise ValueError("ensemble model archive must contain candidates 02 and 03")
    for candidate in CANDIDATES:
        group = value[candidate]
        if not isinstance(group, dict) or set(group) != {"direct", "hurdle"}:
            raise ValueError(f"invalid ensemble model group for candidate {candidate}")
        direct = group["direct"]
        hurdle = group["hurdle"]
        if (
            not isinstance(direct, PredictionFamilyModel)
            or direct.family != "direct_ridge"
            or direct.candidate != candidate
            or not isinstance(direct.baseline, SequentialRidgeModel)
            or not isinstance(direct.enhanced, SequentialRidgeModel)
        ):
            raise ValueError(f"invalid frozen direct model for candidate {candidate}")
        if (
            not isinstance(hurdle, PredictionFamilyModel)
            or hurdle.family != "hurdle"
            or hurdle.candidate != candidate
            or not isinstance(hurdle.baseline, SequentialRidgeModel)
            or not isinstance(hurdle.enhanced, HurdleModel)
        ):
            raise ValueError(f"invalid frozen hurdle model for candidate {candidate}")
    return value


def load_frozen_models(path: Path) -> dict[str, dict[str, PredictionFamilyModel]]:
    with path.open("rb") as handle:
        return _validate_frozen_models(pickle.load(handle))


def _extract_frozen_models(
    all_models: dict[str, dict[str, PredictionFamilyModel]],
) -> dict[str, dict[str, PredictionFamilyModel]]:
    return {
        candidate: {
            "direct": all_models["direct_ridge"][candidate],
            "hurdle": all_models["hurdle"][candidate],
        }
        for candidate in CANDIDATES
    }


def verify_failed_pilot_source(
    design_directory: Path, pilot_directory: Path
) -> dict[str, Any]:
    """Verify the stopped pilot without retroactively authorizing its confirmation."""

    design = verify_design(design_directory)
    pilot = pilot_directory.resolve()
    checks = verify_checksums(pilot)
    seal = json.loads((pilot / "pilot_seal.json").read_text(encoding="utf-8"))
    identifier = seal.pop("pilot_seal_id")
    if seal.get("format") != PILOT_FORMAT or _canonical_digest(seal) != identifier:
        raise ValueError("invalid stopped-pilot seal")
    seal["pilot_seal_id"] = identifier
    if seal["design_registration_id"] != design["registration_id"]:
        raise ValueError("stopped pilot references a different design")
    if seal["source_hashes"] != prediction_source_hashes():
        raise ValueError("registered pilot scientific source changed")
    selection = seal.get("selection", {})
    if (
        seal.get("status") != "stopped_before_confirmation"
        or selection.get("passed") is not False
        or selection.get("selected_family") is not None
    ):
        raise ValueError("source is not the expected failed-selection pilot")
    manifest = json.loads((pilot / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("pilot_seal_id") != identifier
        or manifest.get("confirmation_authorized") is not False
        or manifest.get("selected_family") is not None
    ):
        raise ValueError("pilot manifest does not preserve the registered stop")
    if not all(item.get("adequate") for item in seal["power"].values()):
        raise ValueError("pilot did not meet its event-support prerequisites")
    replay = seal["replay_audit"]
    if not replay.get("discrete_exact") or not replay.get("digests_exact"):
        raise ValueError("pilot replay was not exact")
    all_models_path = pilot / "all_pilot_models.pkl"
    _load_pilot_family_archive(all_models_path)
    return {
        "design": design,
        "seal": seal,
        "manifest": manifest,
        "checksums_verified": len(checks),
        "pilot_path": str(pilot),
        "design_path": str(design_directory.resolve()),
        "pilot_checksum_digest": sha256_file(pilot / "SHA256SUMS"),
        "design_checksum_digest": sha256_file(
            design_directory.resolve() / "SHA256SUMS"
        ),
        "all_pilot_models_digest": sha256_file(all_models_path),
        "pilot_arrays_digest": sha256_file(pilot / "pilot_arrays.npz"),
        "pilot_states_digest": sha256_file(pilot / "pilot_states.csv"),
    }


def _load_development_features(
    path: Path,
) -> tuple[NDArray[np.str_], NDArray[np.str_], PredictionRawFeatures]:
    with np.load(path, allow_pickle=False) as archive:
        state_ids = np.asarray(archive["state_ids"]).astype(str)
        candidates = np.asarray(archive["candidates"]).astype(str)
        raw = PredictionRawFeatures(
            h10=np.asarray(archive["h10"], dtype=np.float64).copy(),
            state=np.asarray(archive["state"], dtype=np.float64).copy(),
            beta=np.asarray(archive["beta"], dtype=np.float64).copy(),
            interaction=np.asarray(archive["interaction"], dtype=np.float64).copy(),
            dynamics=np.asarray(archive["dynamics"], dtype=np.float64).copy(),
        )
    if state_ids.size != candidates.size or raw.h10.shape[0] != state_ids.size:
        raise ValueError("pilot feature archive has inconsistent state counts")
    if set(candidates.tolist()) != set(CANDIDATES):
        raise ValueError("pilot feature archive has unexpected candidates")
    return state_ids, candidates, raw


def ensemble_probability(direct: NDArray, hurdle: NDArray) -> FloatArray:
    """Return the frozen arithmetic mean on the probability scale."""

    left = np.asarray(direct, dtype=np.float64)
    right = np.asarray(hurdle, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("direct and hurdle prediction shapes differ")
    if (
        not np.isfinite(left).all()
        or not np.isfinite(right).all()
        or np.any((left < 0.0) | (left > 1.0))
        or np.any((right < 0.0) | (right > 1.0))
    ):
        raise ValueError("constituent predictions must be finite probabilities")
    return np.clip(0.5 * left + 0.5 * right, 1e-12, 1.0 - 1e-12)


def _predict_candidate(
    models: dict[str, PredictionFamilyModel], raw: PredictionRawFeatures
) -> dict[str, FloatArray]:
    direct = models["direct"].predict(raw)
    hurdle = models["hurdle"].predict(raw)
    if not np.array_equal(direct["h10"], hurdle["h10"]):
        error = float(np.max(np.abs(direct["h10"] - hurdle["h10"])))
        raise ValueError(f"frozen h10 baselines differ (maximum error {error})")
    return {
        "h10": direct["h10"],
        "direct": direct["enhanced"],
        "hurdle": hurdle["enhanced"],
        "ensemble": ensemble_probability(direct["enhanced"], hurdle["enhanced"]),
    }


def _aligned_predictions(
    models: dict[str, dict[str, PredictionFamilyModel]],
    raw: PredictionRawFeatures,
    candidates: NDArray[np.str_],
) -> dict[str, FloatArray]:
    output = {
        name: np.empty(candidates.size, dtype=np.float64)
        for name in ("h10", "direct", "hurdle", "ensemble")
    }
    for candidate in CANDIDATES:
        selected = candidates == candidate
        predicted = _predict_candidate(models[candidate], raw.selected(selected))
        for name, values in predicted.items():
            output[name][selected] = values
    return output


def _prediction_digest(predictions: dict[str, FloatArray]) -> str:
    digest = hashlib.sha256()
    for name in ("h10", "direct", "hurdle", "ensemble"):
        digest.update(name.encode("ascii"))
        digest.update(np.ascontiguousarray(predictions[name]).tobytes())
    return digest.hexdigest()


def _oof_motivation(path: Path) -> dict[str, Any]:
    table = pd.read_csv(path, dtype={"candidate": str}, float_precision="round_trip")
    table["candidate"] = table["candidate"].str.zfill(2)
    direct_h10 = table["prediction_direct_ridge_h10"].to_numpy(dtype=np.float64)
    hurdle_h10 = table["prediction_hurdle_h10"].to_numpy(dtype=np.float64)
    if not np.array_equal(direct_h10, hurdle_h10):
        raise ValueError("pilot out-of-fold h10 predictions differ by family")
    ensemble = ensemble_probability(
        table["prediction_direct_ridge_enhanced"].to_numpy(dtype=np.float64),
        table["prediction_hurdle_enhanced"].to_numpy(dtype=np.float64),
    )
    rows: list[dict[str, Any]] = []
    losses: list[float] = []
    for candidate in CANDIDATES:
        selected = table["candidate"].to_numpy() == candidate
        losses.append(
            float(
                _state_log_loss(
                    table.loc[selected, "q_strict_all"].to_numpy(dtype=np.float64),
                    ensemble[selected],
                ).mean()
            )
        )
        for half in ("A", "B"):
            q = table.loc[selected, f"q_strict_{half}"].to_numpy(dtype=np.float64)
            gain = float(
                (
                    _state_log_loss(q, direct_h10[selected])
                    - _state_log_loss(q, ensemble[selected])
                ).mean()
            )
            rows.append({"candidate": candidate, "half": half, "log_loss_gain": gain})
    return {
        "status": "post_hoc_development_evidence_only",
        "candidate_equal_log_loss": float(np.mean(losses)),
        "four_cell_log_loss_gains": rows,
        "cannot_rescue_failed_pilot": True,
    }


def _write_registration_report(
    path: Path, payload: dict[str, Any], audit: dict[str, Any]
) -> None:
    motivation = audit["pilot_oof_motivation"]
    lines = [
        "# Direct-plus-hurdle ensemble registration",
        "",
        "This is a new prospective confirmation, not a continuation or rescue of the failed pilot registration.",
        "",
        "The exact frozen predictor is `0.5 × direct + 0.5 × hurdle` on the probability scale. Both candidate-specific direct and hurdle models, all preprocessing, and the common h10 comparator come from the checksum-sealed pilot. No fitting or recalibration exists in the confirmation path.",
        "",
        f"Registration ID: `{payload['registration_id']}`",
        f"Source pilot seal: `{payload['source_pilot']['pilot_seal_id']}` (`stopped_before_confirmation`)",
        f"New matrices: **{CONFIRMATION_MATRICES}**",
        f"New primary futures: **{CONFIRMATION_MATRICES * len(CANDIDATES) * len(LANDMARKS) * BRANCHES:,}**",
        "Full exact replay: **required**.",
        "",
        "The motivating pilot ensemble result is explicitly post-hoc and developmental:",
        "",
        f"- Candidate-equal out-of-fold loss: `{motivation['candidate_equal_log_loss']:.12f}`.",
    ]
    for row in motivation["four_cell_log_loss_gains"]:
        lines.append(
            f"- Candidate {row['candidate']} half {row['half']} developmental gain: "
            f"`{row['log_loss_gain']:.12f}`."
        )
    lines.extend(
        [
            "",
            "The prospective claim succeeds only if the ensemble beats h10 in all four candidate-by-half cells, with a positive gain, a positive whole-matrix bootstrap lower bound, and Holm-adjusted whole-matrix randomization `p < 0.05` in every cell.",
            "",
            "Constituent, Brier, rank, and incidence results are secondary and cannot rescue failure.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def prepare_registration(
    design_directory: Path, pilot_directory: Path, output_directory: Path
) -> None:
    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {output_directory}")
    seed_domains = _seed_domains()
    if len(set(seed_domains.values())) != len(seed_domains):
        raise ValueError("ensemble confirmation seed-domain collision")
    source = verify_failed_pilot_source(design_directory, pilot_directory)
    pilot = pilot_directory.resolve()
    all_models = _load_pilot_family_archive(pilot / "all_pilot_models.pkl")
    frozen = _extract_frozen_models(all_models)
    state_ids, candidates, raw = _load_development_features(pilot / "pilot_arrays.npz")
    before = _aligned_predictions(frozen, raw, candidates)

    with _atomic_destination(output_directory) as output:
        with (output / "frozen_ensemble_models.pkl").open("wb") as handle:
            pickle.dump(frozen, handle, protocol=5)
        reloaded = load_frozen_models(output / "frozen_ensemble_models.pkl")
        after = _aligned_predictions(reloaded, raw, candidates)
        errors = {
            name: float(np.max(np.abs(before[name] - after[name]))) for name in before
        }
        if any(value != 0.0 for value in errors.values()):
            raise ValueError("frozen ensemble model round trip changed predictions")
        if not np.array_equal(
            after["ensemble"], ensemble_probability(after["direct"], after["hurdle"])
        ):
            raise ValueError("ensemble archive does not implement the frozen formula")
        np.savez_compressed(
            output / "development_predictions.npz",
            state_ids=state_ids,
            candidates=candidates,
            **after,
        )
        audit: dict[str, Any] = {
            "source_pilot_status": source["seal"]["status"],
            "source_pilot_selection_passed": source["seal"]["selection"]["passed"],
            "source_pilot_failure_preserved": True,
            "states_repredicted": int(state_ids.size),
            "baseline_direct_hurdle_exact": True,
            "ensemble_formula_probability_scale_exact": True,
            "portable_prediction_maximum_absolute_errors": errors,
            "portable_predictions_bit_exact": all(
                value == 0.0 for value in errors.values()
            ),
            "prediction_digest": _prediction_digest(after),
            "models": {
                candidate: {
                    "direct": model_summary(frozen[candidate]["direct"]),
                    "hurdle": model_summary(frozen[candidate]["hurdle"]),
                }
                for candidate in CANDIDATES
            },
            "pilot_oof_motivation": _oof_motivation(pilot / "pilot_states.csv"),
            "confirmation_operations": {
                "model_refitting": False,
                "preprocessing_refitting": False,
                "recalibration": False,
                "regularization_search": False,
                "weight_fitting": False,
                "family_switching": False,
                "nonlinear_fallback": False,
            },
        }
        (output / "development_audit.json").write_text(
            json.dumps(_json_ready(audit), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        protocol = _protocol()
        (output / "protocol.json").write_text(
            json.dumps(_json_ready(protocol), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        source_pilot = {
            "pilot_path": source["pilot_path"],
            "design_path": source["design_path"],
            "pilot_seal_id": source["seal"]["pilot_seal_id"],
            "pilot_status": source["seal"]["status"],
            "pilot_selection_passed": source["seal"]["selection"]["passed"],
            "design_registration_id": source["design"]["registration_id"],
            "pilot_checksum_digest": source["pilot_checksum_digest"],
            "design_checksum_digest": source["design_checksum_digest"],
            "all_pilot_models_digest": source["all_pilot_models_digest"],
            "pilot_arrays_digest": source["pilot_arrays_digest"],
            "pilot_states_digest": source["pilot_states_digest"],
        }
        payload: dict[str, Any] = {
            "format": REGISTRATION_FORMAT,
            "status": "sealed_before_confirmation_matrix_generation",
            "source_pilot": source_pilot,
            "protocol_id": protocol["protocol_id"],
            "protocol_digest": sha256_file(output / "protocol.json"),
            "frozen_models_digest": sha256_file(output / "frozen_ensemble_models.pkl"),
            "development_predictions_digest": sha256_file(
                output / "development_predictions.npz"
            ),
            "development_audit_digest": sha256_file(output / "development_audit.json"),
            "development_prediction_digest": audit["prediction_digest"],
            "source_hashes": _source_hashes(),
            "seed_domains": seed_domains,
            "all_seed_domains_unique": True,
            "confirmation_count": 1,
        }
        payload["registration_id"] = _canonical_digest(payload)
        (output / "registration.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_registration_report(output / "REGISTRATION.md", payload, audit)
        write_checksums(output)
    verify_registration(output_directory)
    print(f"Ensemble confirmation registered at {output_directory}", flush=True)


def _verify_development_predictions(
    registration: Path,
    pilot: Path,
    models: dict[str, dict[str, PredictionFamilyModel]],
    expected_digest: str,
) -> dict[str, Any]:
    state_ids, candidates, raw = _load_development_features(pilot / "pilot_arrays.npz")
    predicted = _aligned_predictions(models, raw, candidates)
    digest = _prediction_digest(predicted)
    if digest != expected_digest:
        raise ValueError("frozen development prediction digest changed")
    with np.load(
        registration / "development_predictions.npz", allow_pickle=False
    ) as archive:
        if not np.array_equal(archive["state_ids"].astype(str), state_ids):
            raise ValueError("registered development state IDs changed")
        if not np.array_equal(archive["candidates"].astype(str), candidates):
            raise ValueError("registered development candidates changed")
        for name, values in predicted.items():
            if not np.array_equal(archive[name], values):
                raise ValueError(f"registered development predictions changed: {name}")
    return {
        "states": int(state_ids.size),
        "prediction_digest": digest,
        "all_arrays_exact": True,
    }


def verify_registration(directory: Path) -> dict[str, Any]:
    directory = directory.resolve()
    verify_checksums(directory)
    payload = json.loads((directory / "registration.json").read_text(encoding="utf-8"))
    identifier = payload.pop("registration_id")
    if (
        payload.get("format") != REGISTRATION_FORMAT
        or payload.get("status") != "sealed_before_confirmation_matrix_generation"
        or _canonical_digest(payload) != identifier
    ):
        raise ValueError("invalid ensemble confirmation registration")
    payload["registration_id"] = identifier
    if payload["source_hashes"] != _source_hashes():
        changed = [
            name
            for name, digest in payload["source_hashes"].items()
            if _source_hashes().get(name) != digest
        ]
        raise ValueError(f"registered ensemble source changed: {changed}")
    if (
        payload["seed_domains"] != _seed_domains()
        or not payload["all_seed_domains_unique"]
        or len(set(payload["seed_domains"].values())) != len(payload["seed_domains"])
    ):
        raise ValueError("ensemble seed-domain contract changed")
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    if (
        protocol != json.loads(json.dumps(_json_ready(_protocol())))
        or payload["protocol_id"] != protocol["protocol_id"]
        or payload["protocol_digest"] != sha256_file(directory / "protocol.json")
    ):
        raise ValueError("ensemble protocol changed after registration")
    for key, filename in (
        ("frozen_models_digest", "frozen_ensemble_models.pkl"),
        ("development_predictions_digest", "development_predictions.npz"),
        ("development_audit_digest", "development_audit.json"),
    ):
        if payload[key] != sha256_file(directory / filename):
            raise ValueError(f"registered artifact digest changed: {filename}")
    source_record = payload["source_pilot"]
    source = verify_failed_pilot_source(
        Path(source_record["design_path"]), Path(source_record["pilot_path"])
    )
    observed_source = {
        "pilot_path": source["pilot_path"],
        "design_path": source["design_path"],
        "pilot_seal_id": source["seal"]["pilot_seal_id"],
        "pilot_status": source["seal"]["status"],
        "pilot_selection_passed": source["seal"]["selection"]["passed"],
        "design_registration_id": source["design"]["registration_id"],
        "pilot_checksum_digest": source["pilot_checksum_digest"],
        "design_checksum_digest": source["design_checksum_digest"],
        "all_pilot_models_digest": source["all_pilot_models_digest"],
        "pilot_arrays_digest": source["pilot_arrays_digest"],
        "pilot_states_digest": source["pilot_states_digest"],
    }
    if source_record != observed_source:
        raise ValueError("registered source-pilot record changed")
    models = load_frozen_models(directory / "frozen_ensemble_models.pkl")
    portable = _verify_development_predictions(
        directory,
        Path(source_record["pilot_path"]),
        models,
        payload["development_prediction_digest"],
    )
    audit = json.loads(
        (directory / "development_audit.json").read_text(encoding="utf-8")
    )
    if (
        not audit.get("source_pilot_failure_preserved")
        or not audit.get("baseline_direct_hurdle_exact")
        or not audit.get("ensemble_formula_probability_scale_exact")
        or not audit.get("portable_predictions_bit_exact")
        or any(audit["confirmation_operations"].values())
    ):
        raise ValueError("development audit does not certify the frozen ensemble")
    payload["portable_verification"] = portable
    return payload


def score_confirmation(
    cases: list[PredictionCase],
    strict_labels: NDArray[np.int8],
    predictions: dict[str, dict[str, FloatArray]],
    development_power: dict[str, dict[str, Any]],
    bootstrap_repetitions: int = BOOTSTRAP_REPETITIONS,
    randomization_repetitions: int = RANDOMIZATION_REPETITIONS,
) -> dict[str, Any]:
    """Evaluate the four frozen transfer cells without fitting any parameter."""

    if strict_labels.shape != (len(cases), BRANCHES):
        raise ValueError(
            "strict-label array does not match the registered branch count"
        )
    ids = _matrix_ids(cases)
    confirmation_power = _power(strict_labels, cases)
    rows: list[dict[str, Any]] = []
    candidates: dict[str, Any] = {}
    for candidate in CANDIDATES:
        selected = _candidate_mask(cases, candidate)
        values = strict_labels[selected]
        candidate_ids = ids[selected]
        expected = int(selected.sum())
        group = predictions[candidate]
        if set(group) != {"h10", "direct", "hurdle", "ensemble"}:
            raise ValueError(f"invalid prediction set for candidate {candidate}")
        if any(np.asarray(item).shape != (expected,) for item in group.values()):
            raise ValueError(f"prediction length mismatch for candidate {candidate}")
        if not np.array_equal(
            group["ensemble"], ensemble_probability(group["direct"], group["hurdle"])
        ):
            raise ValueError(f"ensemble formula mismatch for candidate {candidate}")
        split = values.shape[1] // 2
        q_all = values.mean(axis=1)
        q_a = values[:, :split].mean(axis=1)
        q_b = values[:, split:].mean(axis=1)
        candidates[candidate] = {
            "states": expected,
            "matrices": int(np.unique(candidate_ids).size),
            "strict_events": int(values.sum()),
            "strict_rate": float(values.mean()),
            "models": {
                name: {
                    "pooled_log_loss": float(_state_log_loss(q_all, prediction).mean()),
                    "pooled_q_brier": float(_state_brier(q_all, prediction).mean()),
                    "mean_prediction": float(np.mean(prediction)),
                    "spearman_by_half": [
                        spearman(prediction, q_a),
                        spearman(prediction, q_b),
                    ],
                    "matrix_centered_spearman_by_half": [
                        centered_spearman(prediction, q_a, candidate_ids),
                        centered_spearman(prediction, q_b, candidate_ids),
                    ],
                }
                for name, prediction in group.items()
            },
        }
        for half, q in (("A", q_a), ("B", q_b)):
            seed_parts = (candidate, half)
            gain, interval = _paired_gain(
                q,
                group["h10"],
                group["ensemble"],
                candidate_ids,
                _state_log_loss,
                bootstrap_repetitions,
                np.random.default_rng(
                    derive_seed(
                        BOOTSTRAP_MASTER_SEED,
                        "ensemble_confirmation.log_loss",
                        *seed_parts,
                    )
                ),
            )
            p_value = paired_matrix_randomization_p(
                q,
                group["h10"],
                group["ensemble"],
                candidate_ids,
                randomization_repetitions,
                np.random.default_rng(
                    derive_seed(
                        RANDOMIZATION_MASTER_SEED,
                        "ensemble_confirmation.randomization",
                        *seed_parts,
                    )
                ),
            )
            rows.append(
                {
                    "candidate": candidate,
                    "half": half,
                    "log_loss_gain": gain,
                    "log_loss_gain_ci95": interval,
                    "randomization_p_raw": p_value,
                }
            )
    adjusted = holm_adjust([row["randomization_p_raw"] for row in rows])
    for row, value in zip(rows, adjusted):
        row["randomization_p_holm"] = value
        row["positive_gain"] = bool(row["log_loss_gain"] > 0.0)
        row["positive_bootstrap_lower_bound"] = bool(row["log_loss_gain_ci95"][0] > 0.0)
        row["holm_significant"] = bool(value < 0.05)
        row["passes_transfer_gate"] = bool(
            row["positive_gain"]
            and row["positive_bootstrap_lower_bound"]
            and row["holm_significant"]
        )
    power_adequate = all(
        development_power[candidate]["adequate"]
        and confirmation_power[candidate]["adequate"]
        for candidate in CANDIDATES
    )
    transfer_gates_pass = bool(
        rows and all(row["passes_transfer_gate"] for row in rows)
    )
    return {
        "primary_tests": rows,
        "candidates": candidates,
        "development_power": development_power,
        "confirmation_power": confirmation_power,
        "power_adequate": power_adequate,
        "all_four_transfer_gates_pass": transfer_gates_pass,
        "primary_prediction_supported": bool(power_adequate and transfer_gates_pass),
        "family_size": 4,
        "predictor": "0.5 * direct + 0.5 * hurdle on probability scale",
        "decision_rule": (
            "positive ensemble-over-h10 log-loss gain, whole-matrix bootstrap "
            "95% lower bound > 0, and Holm-adjusted whole-matrix randomization "
            "p < 0.05 in both candidates and both fixed branch halves"
        ),
        "fitting_or_recalibration_performed": False,
    }


def _secondary_descriptives(
    cases: list[PredictionCase],
    labels: dict[str, NDArray[np.int8]],
    predictions: dict[str, dict[str, FloatArray]],
) -> dict[str, Any]:
    ids = _matrix_ids(cases)
    output: dict[str, Any] = {
        "status": "secondary_no_rescue",
        "constituent_log_loss_gains": [],
        "ensemble_brier_gains": [],
        "endpoint_incidence": {},
    }
    for candidate in CANDIDATES:
        selected = _candidate_mask(cases, candidate)
        candidate_ids = ids[selected]
        output["endpoint_incidence"][candidate] = {
            name: {
                "events": int(values[selected].sum()),
                "rate": float(values[selected].mean()),
            }
            for name, values in labels.items()
        }
        strict = labels["strict"][selected]
        split = strict.shape[1] // 2
        for half, q in (
            ("A", strict[:, :split].mean(axis=1)),
            ("B", strict[:, split:].mean(axis=1)),
        ):
            for name in ("direct", "hurdle"):
                gain = float(
                    (
                        _state_log_loss(q, predictions[candidate]["h10"])
                        - _state_log_loss(q, predictions[candidate][name])
                    ).mean()
                )
                output["constituent_log_loss_gains"].append(
                    {"candidate": candidate, "half": half, "model": name, "gain": gain}
                )
            brier, interval = _paired_gain(
                q,
                predictions[candidate]["h10"],
                predictions[candidate]["ensemble"],
                candidate_ids,
                _state_brier,
                BOOTSTRAP_REPETITIONS,
                np.random.default_rng(
                    derive_seed(
                        BOOTSTRAP_MASTER_SEED,
                        "ensemble_confirmation.secondary_brier",
                        candidate,
                        half,
                    )
                ),
            )
            output["ensemble_brier_gains"].append(
                {
                    "candidate": candidate,
                    "half": half,
                    "gain": brier,
                    "ci95": interval,
                }
            )
    return output


def _prepare_campaign(
    work_directory: Path, registration: dict[str, Any], output_directory: Path
) -> None:
    work_directory.mkdir(parents=True, exist_ok=True)
    contract_path = work_directory / "ensemble_campaign_contract.json"
    expected = {
        "format": CAMPAIGN_FORMAT,
        "registration_id": registration["registration_id"],
        "source_hashes": _source_hashes(),
        "experiment": _experiment(
            CONFIRMATION_MASTER_SEED, CONFIRMATION_MATRICES
        ).to_dict(),
        "cohort_label": COHORT_LABEL,
        "output_directory": str(output_directory.resolve()),
    }
    if contract_path.exists():
        observed = json.loads(contract_path.read_text(encoding="utf-8"))
        if observed != json.loads(json.dumps(_json_ready(expected))):
            raise ValueError("ensemble campaign checkpoint contract changed")
        return
    if any(work_directory.iterdir()):
        raise ValueError("unregistered files in new ensemble campaign work directory")
    contract_path.write_text(
        json.dumps(_json_ready(expected), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _confirmation_predictions(
    cases: list[PredictionCase],
    raw: PredictionRawFeatures,
    models: dict[str, dict[str, PredictionFamilyModel]],
) -> dict[str, dict[str, FloatArray]]:
    output: dict[str, dict[str, FloatArray]] = {}
    for candidate in CANDIDATES:
        selected = _candidate_mask(cases, candidate)
        output[candidate] = _predict_candidate(
            models[candidate], raw.selected(selected)
        )
    return output


def _write_state_table(
    path: Path,
    cases: list[PredictionCase],
    strict: NDArray[np.int8],
    predictions: dict[str, dict[str, FloatArray]],
) -> None:
    local_indices = {candidate: 0 for candidate in CANDIDATES}
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        local = local_indices[case.candidate]
        local_indices[case.candidate] += 1
        group = predictions[case.candidate]
        rows.append(
            {
                "state_id": case.state_id,
                "candidate": case.candidate,
                "matrix_id": case.matrix_id,
                "landmark": case.landmark,
                "q_strict_all": float(strict[index].mean()),
                "q_strict_A": float(strict[index, : BRANCHES // 2].mean()),
                "q_strict_B": float(strict[index, BRANCHES // 2 :].mean()),
                "prediction_h10": float(group["h10"][local]),
                "prediction_direct": float(group["direct"][local]),
                "prediction_hurdle": float(group["hurdle"][local]),
                "prediction_ensemble": float(group["ensemble"][local]),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.17g")


def _read_state_predictions(path: Path) -> dict[str, dict[str, FloatArray]]:
    table = pd.read_csv(path, dtype={"candidate": str}, float_precision="round_trip")
    table["candidate"] = table["candidate"].str.zfill(2)
    return {
        candidate: {
            name: table.loc[
                table["candidate"] == candidate, f"prediction_{name}"
            ].to_numpy(dtype=np.float64)
            for name in ("h10", "direct", "hurdle", "ensemble")
        }
        for candidate in CANDIDATES
    }


def _write_confirmation_report(
    path: Path,
    primary: dict[str, Any],
    replay: dict[str, Any],
    registration: dict[str, Any],
) -> None:
    lines = [
        "# Prospective direct-plus-hurdle ensemble confirmation",
        "",
        f"Primary prediction supported: **{primary['primary_prediction_supported']}**.",
        "",
        "This is the one separately registered confirmation of the pilot-developed `0.5 × direct + 0.5 × hurdle` probability ensemble. The earlier pilot remains a registered failure and is not reinterpreted.",
        "",
        "| Candidate | Half | Log-loss gain | 95% matrix-bootstrap CI | Raw p | Holm p | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary["primary_tests"]:
        interval = row["log_loss_gain_ci95"]
        lines.append(
            f"| {row['candidate']} | {row['half']} | {row['log_loss_gain']:.8f} | "
            f"[{interval[0]:.8f}, {interval[1]:.8f}] | "
            f"{row['randomization_p_raw']:.6g} | {row['randomization_p_holm']:.6g} | "
            f"{row['passes_transfer_gate']} |"
        )
    lines.extend(
        [
            "",
            f"Power prerequisite adequate: **{primary['power_adequate']}**.",
            f"Exact full replay: **{replay['digests_exact']}** (maximum continuous error `{replay['maximum_continuous_absolute_error']}`).",
            f"Registration ID: `{registration['registration_id']}`.",
            "",
            "The primary result concerns prospective prediction of a strict break-and-distinct-renewal event beyond h10 history/clocks. It does not establish causal control, attractor switching, recurrence, or origin-of-life realism. Constituent, Brier, rank, and incidence descriptives cannot rescue a failed four-cell primary gate.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_confirmation(
    registration_directory: Path,
    output_directory: Path,
    workers: int,
    work_directory: Path | None = None,
) -> None:
    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {output_directory}")
    registration = verify_registration(registration_directory)
    models = load_frozen_models(
        registration_directory.resolve() / "frozen_ensemble_models.pkl"
    )
    experiment = _experiment(CONFIRMATION_MASTER_SEED, CONFIRMATION_MATRICES)
    work = (
        work_directory.resolve()
        if work_directory is not None
        else output_directory.with_name(f".{output_directory.name}.work")
    )
    _prepare_campaign(work, registration, output_directory)
    _campaign_status(
        work, "ensemble_confirmation", "building_trajectories_and_features"
    )
    print(
        "[ensemble 1/9] Generating 200 entirely new matrices and 2,000 states",
        flush=True,
    )
    with threadpool_limits(limits=1):
        cases = build_prediction_cohort(
            experiment, COHORT_LABEL, experiment.confirmation
        )
        raw = extract_prediction_features(cases, experiment)
    predictions = _confirmation_predictions(cases, raw, models)
    print("[ensemble 2/9] Shooting 256,000 untouched F32 futures", flush=True)
    _campaign_status(work, "ensemble_confirmation", "shooting_futures")
    batches = run_prediction_branches(
        cases,
        experiment,
        BRANCHES,
        workers,
        "ensemble-confirm",
        checkpoint_directory=work / "generate",
    )
    print("[ensemble 3/9] Replaying all 256,000 futures", flush=True)
    _campaign_status(work, "ensemble_confirmation", "replaying_futures")
    replay_batches = run_prediction_branches(
        cases,
        experiment,
        BRANCHES,
        workers,
        "ensemble-confirm-replay",
        checkpoint_directory=work / "replay",
    )
    replay = replay_audit(batches, replay_batches)
    labels = _labels(batches)
    development_power = json.loads(
        (
            Path(registration["source_pilot"]["pilot_path"]) / "pilot_seal.json"
        ).read_text(encoding="utf-8")
    )["power"]
    _campaign_status(work, "ensemble_confirmation", "computing_frozen_inference")
    primary = score_confirmation(
        cases, labels["strict"], predictions, development_power
    )
    secondary = _secondary_descriptives(cases, labels, predictions)

    with _atomic_destination(output_directory) as output:
        print(
            "[ensemble 4/9] Writing complete branch, state, and feature artifacts",
            flush=True,
        )
        _write_branch_tables(output, "confirmation", cases, batches)
        _save_arrays(output / "confirmation_arrays.npz", cases, raw, batches)
        _write_state_table(
            output / "confirmation_states.csv", cases, labels["strict"], predictions
        )
        for name, value in (
            ("primary_metrics.json", primary),
            ("secondary_descriptives.json", secondary),
            ("replay_audit.json", replay),
        ):
            (output / name).write_text(
                json.dumps(_json_ready(value), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        print(
            "[ensemble 5/9] Recomputing the primary decision from saved files",
            flush=True,
        )
        restored_labels = _strict_labels_from_branch_table(
            output / "confirmation_branches.csv.gz", cases
        )
        if not np.array_equal(restored_labels, labels["strict"]):
            raise ValueError("round-trip strict labels changed")
        restored_predictions = _read_state_predictions(
            output / "confirmation_states.csv"
        )
        readback = score_confirmation(
            cases, restored_labels, restored_predictions, development_power
        )
        if _json_ready(readback) != _json_ready(primary):
            raise ValueError("round-trip primary metrics changed")
        (output / "readback_audit.json").write_text(
            json.dumps(
                {
                    "branch_labels_exact": True,
                    "prediction_float_precision": "round_trip",
                    "primary_metrics_exact": True,
                    "fitting_or_recalibration_performed": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        _write_confirmation_report(
            output / "CONFIRMATION_REPORT.md", primary, replay, registration
        )
        manifest = {
            "format": CONFIRMATION_FORMAT,
            "registration_id": registration["registration_id"],
            "source_pilot_seal_id": registration["source_pilot"]["pilot_seal_id"],
            "source_pilot_status": "stopped_before_confirmation",
            "predictor": "0.5 * direct + 0.5 * hurdle on probability scale",
            "matrices": CONFIRMATION_MATRICES,
            "states": len(cases),
            "primary_futures": len(cases) * BRANCHES,
            "replay_futures": len(cases) * BRANCHES,
            "primary_prediction_supported": primary["primary_prediction_supported"],
            "all_four_transfer_gates_pass": primary["all_four_transfer_gates_pass"],
            "power_adequate": primary["power_adequate"],
            "replay_exact": replay["digests_exact"],
            "no_refitting_or_recalibration": True,
            "runtime": _runtime_manifest(),
            "checkpoint_audit": {
                "work_directory": str(work),
                "campaign_contract_digest": sha256_file(
                    work / "ensemble_campaign_contract.json"
                ),
                "generation_contract_digest": sha256_file(
                    work / "generate" / "checkpoint_contract.json"
                ),
                "replay_contract_digest": sha256_file(
                    work / "replay" / "checkpoint_contract.json"
                ),
                "resumable_per_state": True,
            },
            "claim_boundary": "prediction, not control or regime switching",
        }
        (output / "manifest.json").write_text(
            json.dumps(_json_ready(manifest), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("[ensemble 6/9] Sealing confirmation artifacts", flush=True)
        write_checksums(output)
    verify_checksums(output_directory)
    _campaign_status(work, "ensemble_confirmation", "sealed_complete")
    print("[ensemble 7/9] Confirmation sealed and checksum-verified", flush=True)
    print(
        "[ensemble 8/9] Failed pilot and all earlier cohorts remain unchanged",
        flush=True,
    )
    print(f"[ensemble 9/9] Results: {output_directory}", flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prospective direct-plus-hurdle ensemble confirmation"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument(
        "--design",
        type=Path,
        default=Path("results/regime_prediction_registration"),
    )
    prepare.add_argument(
        "--pilot", type=Path, default=Path("results/regime_prediction_pilot")
    )
    prepare.add_argument(
        "--output",
        type=Path,
        default=Path("results/regime_ensemble_registration"),
    )
    confirm = commands.add_parser("confirm")
    confirm.add_argument(
        "--registration",
        type=Path,
        default=Path("results/regime_ensemble_registration"),
    )
    confirm.add_argument(
        "--output",
        type=Path,
        default=Path("results/regime_ensemble_confirmation"),
    )
    confirm.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 14))
    confirm.add_argument("--work-dir", type=Path, default=None)
    verify = commands.add_parser("verify")
    verify.add_argument(
        "--registration",
        type=Path,
        default=Path("results/regime_ensemble_registration"),
    )
    status = commands.add_parser("status")
    status.add_argument("--work-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    if arguments.command == "prepare":
        prepare_registration(arguments.design, arguments.pilot, arguments.output)
    elif arguments.command == "confirm":
        if arguments.workers < 1:
            raise ValueError("workers must be positive")
        run_confirmation(
            arguments.registration,
            arguments.output,
            arguments.workers,
            arguments.work_dir,
        )
    elif arguments.command == "verify":
        payload = verify_registration(arguments.registration)
        print(
            json.dumps(
                {
                    "registration_id": payload["registration_id"],
                    "status": payload["status"],
                    "portable_verification": payload["portable_verification"],
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif arguments.command == "status":
        print(
            json.dumps(
                read_checkpoint_status(arguments.work_dir), indent=2, sort_keys=True
            )
        )
    else:  # pragma: no cover
        raise AssertionError(arguments.command)


if __name__ == "__main__":
    main()
