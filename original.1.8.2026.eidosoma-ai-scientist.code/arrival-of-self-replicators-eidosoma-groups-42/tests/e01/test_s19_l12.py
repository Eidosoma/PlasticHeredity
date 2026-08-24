from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from scripts.e01 import run_s19_l12 as l12


def test_preregistration_is_analysis_only_and_prohibits_scientific_execution() -> None:
    config = l12.read_config()
    assert config["researchStepId"] == "S19-L12"
    assert config["mode"] == "ANALYSIS_ONLY_FORENSIC_SOURCE_AND_EVIDENCE_AUDIT"
    assert config["resources"]["gpuHoursMaximum"] == 0
    assert all(config["prohibited"].values())


def test_all_registered_paper_components_are_audited() -> None:
    panels, _ = l12.figure_specs()
    required = set(l12.read_config()["requiredPaperComponents"])
    assert required.issubset(set(panels["panelId"]))
    assert len(panels) == len(required)


def test_statement_registry_covers_methods_and_all_59_claims() -> None:
    registry = l12.build_paper_statement_registry()
    assert len(registry) == 96
    assert registry["paperStatementId"].is_unique
    assert registry["paperStatementId"].str.startswith("CLAIM_E01-C").sum() == 59
    required = {
        "timeUnit", "statisticalUnit", "preprocessing", "estimator", "label",
        "denominator", "aggregation", "statisticalTest", "reportedValue",
        "directlySpecified", "partiallySpecified", "internallyConflicting",
        "absentFromPublicCode", "E01Implementation", "E01Result", "unresolvedFields",
    }
    assert required.issubset(registry.columns)


def test_safe_lattice_enumeration_and_metric_coefficients() -> None:
    atoms = l12.build_atom_registry()
    assert len(atoms) == 16
    assert atoms[["sourceAntichain", "targetAntichain"]].drop_duplicates().shape[0] == 16
    assert atoms["localPhiRWeight"].sum() == 9
    assert atoms["publicEmergenceWeight"].sum() == 3
    assert atoms["paperDisplayedEquationWeight"].value_counts().to_dict() == {-1: 4, 0: 8, 1: 4}
    assert np.array_equal(atoms["paperDisplayedEquationWeight"], atoms["directWholeMinusPartsWeight"])
    assert not np.array_equal(atoms["paperDisplayedEquationWeight"], atoms["localPhiRWeight"])
    assert not np.array_equal(atoms["paperDisplayedEquationWeight"], atoms["publicEmergenceWeight"])


def test_metric_adjudication_is_algebraic_not_outcome_selected() -> None:
    adjudication = l12.metric_identity_adjudication(l12.build_atom_registry())
    assert adjudication["classification"] == "PAPER_METRIC_IDENTITY_INTERNALLY_INCONSISTENT"
    assert adjudication["paperEquationEqualsDirectWms"]
    assert not adjudication["paperEquationEqualsPublicIntegratedIdentity"]
    assert not adjudication["paperEquationEqualsPublicEmergenceIdentity"]
    assert not adjudication["selectionByReplicationAssociation"]


def test_figure5_reconciliation_family_is_complete_and_nonexecuting() -> None:
    frame = l12.figure5_reconciliation_rows()
    required = {
        "class balancing", "per-run balancing", "onset-only target", "full future-state target",
        "generation-level target", "molecular-level target", "run-level accuracy averaging",
        "padding included", "padding masked", "truncation to common length",
        "negative-case enrichment", "stratified sampling", "separately generated prediction data",
        "different label identity",
    }
    assert set(frame["possibility"]) == required


def test_hidden_pipeline_generation_fails_before_concordance_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(l12, "OUT", tmp_path)
    with pytest.raises(RuntimeError, match="before concordance lock"):
        l12.hidden_pipeline_hypotheses()


def test_runner_does_not_import_gard_or_training_modules() -> None:
    tree = ast.parse(Path(l12.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    prohibited_fragments = {"gard", "torch", "tensorflow", "stable_baselines"}
    assert not any(any(fragment in name.lower() for fragment in prohibited_fragments) for name in imported)


def test_public_phirl_dataflow_exposes_completed_fit_dependencies() -> None:
    graph = l12.build_phirl_dataflow()
    assert graph.has_edge("input", "active_filter")
    assert graph.has_edge("fiedler", "means")
    assert graph.has_edge("lattice", "integrated")
    assert graph.has_edge("downward", "emergence")
    assert graph.nodes["gaussian"]["usesCompleteTrajectory"]
    assert graph.nodes["emergence"]["usesCompleteTrajectory"]


def test_concordance_material_rows_use_only_registered_evidence_labels() -> None:
    allowed = set(l12.read_config()["evidenceLabels"])
    rows = l12.material_concordance_rows()
    assert len(rows) == 33
    for row in rows:
        assert set(row["evidenceLabels"].split(";")).issubset(allowed)
