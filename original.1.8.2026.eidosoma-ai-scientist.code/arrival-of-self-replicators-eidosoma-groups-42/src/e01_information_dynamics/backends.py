"""Fail-closed wrappers around the pinned phyid and OmegaID source trees."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .validation import ATOM_IDS, aggregate_means, strict_sample_gate

PHYID_ROOT = Path("/cache/e01_s03/sources/phyid")
OMEGAID_ROOT = Path("/cache/e01_s03/sources/omegaid")
EXPECTED_PHYID_COMMIT = "6c5f2e9d33c985efbdf875d45cb5a2a6a5cdbf44"
EXPECTED_OMEGAID_COMMIT = "7fcf1fa8e288e0634f81423283d2b349ed88440e"


class InformationBackendError(RuntimeError):
    """A backend request violates a frozen branch contract."""


@dataclass(frozen=True, slots=True)
class DecompositionResult:
    """One lossless in-memory decomposition in reported nats."""

    backend_id: str
    kind: Literal["gaussian", "discrete"]
    redundancy: Literal["MMI", "CCS"]
    status: str
    reason: str | None
    atoms: dict[str, np.ndarray] | None
    intermediate_mi: dict[str, np.ndarray] | None
    intermediate_redundancy: dict[str, np.ndarray] | None
    double_redundancy: np.ndarray | None
    sample_gate: dict[str, Any]
    native_units: str
    reported_units: str
    regularization_entered: bool | None

    def means(self) -> dict[str, Any] | None:
        if self.atoms is None or self.intermediate_mi is None:
            return None
        return aggregate_means(self.atoms, self.intermediate_mi)


def _prepend_pinned_paths() -> None:
    """Place the official phyid tree before OmegaID's vendored phyid copy."""

    for path in (str(OMEGAID_ROOT), str(PHYID_ROOT)):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)


def _assert_module_path(module: Any, expected_root: Path, name: str) -> None:
    resolved = Path(module.__file__).resolve()
    if expected_root.resolve() not in resolved.parents:
        raise InformationBackendError(
            f"{name} imported from {resolved}, expected pinned tree {expected_root}."
        )


def _load_phyid() -> tuple[Any, Any]:
    _prepend_pinned_paths()
    calculate = importlib.import_module("phyid.calculate")
    utils = importlib.import_module("phyid.utils")
    _assert_module_path(calculate, PHYID_ROOT, "phyid.calculate")
    _assert_module_path(utils, PHYID_ROOT, "phyid.utils")
    return calculate, utils


def _load_omegaid() -> tuple[Any, Any]:
    _prepend_pinned_paths()
    decomposition = importlib.import_module("omegaid.core.decomposition")
    backend = importlib.import_module("omegaid.utils.backend")
    _assert_module_path(decomposition, OMEGAID_ROOT, "omegaid.core.decomposition")
    _assert_module_path(backend, OMEGAID_ROOT, "omegaid.utils.backend")
    return decomposition, backend


def backend_identity() -> dict[str, str]:
    """Return the resolved source modules used by the wrappers."""

    phyid, _ = _load_phyid()
    omegaid, _ = _load_omegaid()
    return {
        "phyidCalculate": str(Path(phyid.__file__).resolve()),
        "omegaidDecomposition": str(Path(omegaid.__file__).resolve()),
        "phyidCommit": EXPECTED_PHYID_COMMIT,
        "omegaidCommit": EXPECTED_OMEGAID_COMMIT,
    }


def _normalize_result(
    *,
    atoms: dict[str, Any],
    calculations: dict[str, Any],
    kind: Literal["gaussian", "discrete"],
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    np.ndarray,
]:
    if set(atoms) != set(ATOM_IDS):
        raise InformationBackendError(f"Unexpected atom identities: {sorted(atoms)}")
    factor = float(np.log(2.0)) if kind == "discrete" else 1.0
    atom_arrays = {
        key: np.asarray(value, dtype=np.float64) * factor
        for key, value in atoms.items()
    }
    mi_arrays = {
        key: np.asarray(value, dtype=np.float64) * factor
        for key, value in calculations["I_res"].items()
    }
    redundancy_arrays = {
        key: np.asarray(value, dtype=np.float64) * factor
        for key, value in calculations["R_res"].items()
    }
    rtr = np.asarray(calculations["rtr"], dtype=np.float64) * factor
    arrays = [
        *atom_arrays.values(),
        *mi_arrays.values(),
        *redundancy_arrays.values(),
        rtr,
    ]
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise InformationBackendError("Backend returned a nonfinite decomposition.")
    return atom_arrays, mi_arrays, redundancy_arrays, rtr


def run_phyid(
    source: np.ndarray,
    target: np.ndarray,
    *,
    tau: int,
    kind: Literal["gaussian", "discrete"],
    redundancy: Literal["MMI", "CCS"],
) -> DecompositionResult:
    """Run the pinned scalar phyid source after the preregistered sample gate."""

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    gate = strict_sample_gate(source, target, tau=tau, kind=kind)
    if gate["status"] != "ELIGIBLE":
        return DecompositionResult(
            backend_id="pinned_phyid_cpu",
            kind=kind,
            redundancy=redundancy,
            status="INELIGIBLE",
            reason=gate["reason"],
            atoms=None,
            intermediate_mi=None,
            intermediate_redundancy=None,
            double_redundancy=None,
            sample_gate=gate,
            native_units="bits" if kind == "discrete" else "nats",
            reported_units="nats",
            regularization_entered=False,
        )
    calculate, _ = _load_phyid()
    atoms, calculations = calculate.calc_PhiID(
        source,
        target,
        tau,
        kind=kind,
        redundancy=redundancy,
    )
    normalized = _normalize_result(atoms=atoms, calculations=calculations, kind=kind)
    return DecompositionResult(
        backend_id="pinned_phyid_cpu",
        kind=kind,
        redundancy=redundancy,
        status="ELIGIBLE",
        reason=None,
        atoms=normalized[0],
        intermediate_mi=normalized[1],
        intermediate_redundancy=normalized[2],
        double_redundancy=normalized[3],
        sample_gate=gate,
        native_units="bits" if kind == "discrete" else "nats",
        reported_units="nats",
        regularization_entered=False,
    )


def run_omegaid(
    source: np.ndarray,
    target: np.ndarray,
    *,
    tau: int,
    kind: Literal["gaussian", "discrete"],
    redundancy: Literal["MMI", "CCS"],
    backend_name: Literal["numpy", "cupy"],
) -> DecompositionResult:
    """Run pinned OmegaID's scalar 2x2 path under an explicit backend identity."""

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    gate = strict_sample_gate(source, target, tau=tau, kind=kind)
    backend_id = f"pinned_omegaid_{backend_name}"
    if gate["status"] != "ELIGIBLE":
        return DecompositionResult(
            backend_id=backend_id,
            kind=kind,
            redundancy=redundancy,
            status="INELIGIBLE",
            reason=gate["reason"],
            atoms=None,
            intermediate_mi=None,
            intermediate_redundancy=None,
            double_redundancy=None,
            sample_gate=gate,
            native_units="bits" if kind == "discrete" else "nats",
            reported_units="nats",
            regularization_entered=None,
        )
    decomposition, backend = _load_omegaid()
    backend.set_backend(backend_name)
    matrix = np.stack([source, target])
    atoms, calculations = decomposition.calc_phiid_multivariate(
        matrix,
        matrix,
        tau=tau,
        kind=kind,
        redundancy=redundancy,
    )
    if calculations is None:
        raise InformationBackendError("OmegaID unexpectedly entered its doublet path.")
    normalized = _normalize_result(atoms=atoms, calculations=calculations, kind=kind)
    return DecompositionResult(
        backend_id=backend_id,
        kind=kind,
        redundancy=redundancy,
        status="ELIGIBLE",
        reason=None,
        atoms=normalized[0],
        intermediate_mi=normalized[1],
        intermediate_redundancy=normalized[2],
        double_redundancy=normalized[3],
        sample_gate=gate,
        native_units="bits" if kind == "discrete" else "nats",
        reported_units="nats",
        regularization_entered=False,
    )


def compare_decompositions(
    left: DecompositionResult,
    right: DecompositionResult,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    """Compare every local atom and intermediate MI/redundancy vector."""

    if left.status != "ELIGIBLE" or right.status != "ELIGIBLE":
        return {
            "status": "INELIGIBLE",
            "reason": "ONE_OR_BOTH_DECOMPOSITIONS_INELIGIBLE",
            "success": False,
        }
    assert left.atoms is not None and right.atoms is not None
    assert left.intermediate_mi is not None and right.intermediate_mi is not None
    assert left.intermediate_redundancy is not None
    assert right.intermediate_redundancy is not None
    collections = {
        "atom": (left.atoms, right.atoms),
        "mi": (left.intermediate_mi, right.intermediate_mi),
        "redundancy": (
            left.intermediate_redundancy,
            right.intermediate_redundancy,
        ),
    }
    maximum_absolute = 0.0
    maximum_relative = 0.0
    worst_field = None
    local_success = True
    mean_success = True
    for family, (left_values, right_values) in collections.items():
        if set(left_values) != set(right_values):
            raise InformationBackendError(f"{family} field identities differ.")
        for key in sorted(left_values):
            a = np.asarray(left_values[key], dtype=np.float64)
            b = np.asarray(right_values[key], dtype=np.float64)
            if a.shape != b.shape:
                raise InformationBackendError(f"Shape mismatch for {family}.{key}.")
            absolute = np.abs(a - b)
            denominator = np.maximum(
                np.maximum(np.abs(a), np.abs(b)), np.finfo(float).tiny
            )
            relative = absolute / denominator
            candidate = float(np.max(absolute)) if absolute.size else 0.0
            if candidate >= maximum_absolute:
                maximum_absolute = candidate
                maximum_relative = float(np.max(relative)) if relative.size else 0.0
                worst_field = f"{family}.{key}"
            local_success &= bool(
                np.allclose(
                    a,
                    b,
                    atol=absolute_tolerance,
                    rtol=relative_tolerance,
                )
            )
            mean_success &= bool(
                np.isclose(
                    float(np.mean(a)),
                    float(np.mean(b)),
                    atol=absolute_tolerance,
                    rtol=relative_tolerance,
                )
            )
    return {
        "status": "PASS" if local_success and mean_success else "FAIL",
        "success": bool(local_success and mean_success),
        "localSuccess": bool(local_success),
        "meanSuccess": bool(mean_success),
        "maximumAbsoluteError": maximum_absolute,
        "maximumRelativeErrorAtWorstAbsolute": maximum_relative,
        "worstField": worst_field,
        "absoluteTolerance": absolute_tolerance,
        "relativeTolerance": relative_tolerance,
    }
