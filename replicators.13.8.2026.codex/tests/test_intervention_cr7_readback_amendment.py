from __future__ import annotations

import hashlib

from plastic_heredity import intervention_cr7_steering as cr7
from plastic_heredity.intervention_cr7_readback_amendment import (
    ORIGINAL_REGISTRATION,
    ORIGINAL_REGISTRATION_ID,
    reporting_extension_batches,
)


def _artificial_batches() -> tuple[
    list[object], list[cr7.SteeringBatch], list[cr7.SteeringBatch]
]:
    case, _experiment = cr7._artificial_case()
    parent = cr7.SteeringBatch(
        format=cr7.CHECKPOINT_FORMAT,
        registration_id="fixture-registration",
        mode="primary",
        state_id=case.state_id,
        candidate=case.candidate,
        matrix_id=case.matrix_id,
        landmark=case.landmark,
        case_digest=cr7._case_digest(case),
        lineages=(),
    )
    extension = cr7.SteeringBatch(
        format=cr7.CHECKPOINT_FORMAT,
        registration_id="fixture-registration",
        mode="conditional_active_extension",
        state_id=case.state_id,
        candidate=case.candidate,
        matrix_id=case.matrix_id,
        landmark=case.landmark,
        case_digest=hashlib.sha256(
            (cr7._case_digest(case) + cr7.batch_digest(parent)).encode()
        ).hexdigest(),
        lineages=(),
    )
    return [case], [parent], [extension]


def test_cr7_amendment_preserves_original_scientific_registration() -> None:
    registration = cr7.verify_registration(ORIGINAL_REGISTRATION)
    assert registration["registration_id"] == ORIGINAL_REGISTRATION_ID
    assert registration["source_hashes"] == cr7.source_hashes()


def test_reporting_view_changes_only_extension_case_digest() -> None:
    cases, primary, extension = _artificial_batches()
    views = reporting_extension_batches(cases, primary, extension)
    assert len(views) == len(extension) == 1
    for case, source, view in zip(cases, extension, views, strict=True):
        assert view.case_digest == cr7._case_digest(case)
        assert view.case_digest != source.case_digest
        assert view.lineages is source.lineages
        assert view.format == source.format
        assert view.registration_id == source.registration_id
        assert view.mode == source.mode
        assert view.state_id == source.state_id
        assert view.candidate == source.candidate
        assert view.matrix_id == source.matrix_id
        assert view.landmark == source.landmark


def test_original_helper_rejects_parent_bound_extension_before_outcome_read() -> None:
    cases, _primary, extension = _artificial_batches()
    try:
        cr7._lineage_and_matrix_tables([cases[0]], [extension[0]])
    except ValueError as error:
        assert str(error) == "CR7 batch no longer matches its launch state"
    else:  # pragma: no cover
        raise AssertionError("original generic helper unexpectedly accepted extension metadata")


def test_amendment_does_not_replace_scientific_runner() -> None:
    assert reporting_extension_batches is not cr7.extension_summary
    assert cr7.protocol()["operational"]["cr8_and_cr9_not_launched_automatically"]
