from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass
class FrozenStudent:
    name: str
    scaler: StandardScaler
    classifier: LogisticRegression
    pca: PCA | None = None

    def transform(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        scaled = self.scaler.transform(features)
        return self.pca.transform(scaled) if self.pca is not None else scaled

    def predict(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        transformed = self.transform(features)
        return self.classifier.predict_proba(transformed)[:, 1]


@dataclass
class CandidateStudents:
    history: FrozenStudent
    beta: FrozenStudent
    full: FrozenStudent
    prior: float


def _repeat_by_branches(features: NDArray, branch_labels: NDArray) -> tuple[NDArray, NDArray]:
    if branch_labels.ndim != 2 or branch_labels.shape[0] != features.shape[0]:
        raise ValueError("branch labels must be states x branches")
    repeated = np.repeat(features, branch_labels.shape[1], axis=0)
    return repeated, branch_labels.reshape(-1).astype(np.int64)


def _fit_student(
    name: str,
    state_features: NDArray[np.float64],
    branch_labels: NDArray[np.int64],
    c: float,
    pca_components: int | None = None,
) -> FrozenStudent:
    scaler = StandardScaler().fit(state_features)
    scaled_states = scaler.transform(state_features)
    pca = None
    transformed_states = scaled_states
    if pca_components is not None:
        component_count = min(pca_components, *scaled_states.shape)
        pca = PCA(n_components=component_count, svd_solver="full").fit(scaled_states)
        transformed_states = pca.transform(scaled_states)
    train_x, train_y = _repeat_by_branches(transformed_states, branch_labels)
    classifier = LogisticRegression(
        C=c,
        penalty="l2",
        solver="lbfgs",
        max_iter=2_000,
        random_state=0,
    ).fit(train_x, train_y)
    return FrozenStudent(name=name, scaler=scaler, pca=pca, classifier=classifier)


def fit_students(
    state_graph: NDArray[np.float64],
    history: NDArray[np.float64],
    beta: NDArray[np.float64],
    branch_labels: NDArray[np.int64],
    pca_components: int,
    c: float,
) -> CandidateStudents:
    history_student = _fit_student("history", history, branch_labels, c)
    beta_student = _fit_student(
        "beta", beta, branch_labels, c, pca_components=pca_components
    )

    state_scaler = StandardScaler().fit(state_graph)
    scaled_state = state_scaler.transform(state_graph)
    component_count = min(pca_components, *scaled_state.shape)
    state_pca = PCA(n_components=component_count, svd_solver="full").fit(scaled_state)
    state_components = state_pca.transform(scaled_state)
    full_features = np.column_stack((state_components, history))
    full_scaler = StandardScaler().fit(full_features)
    transformed_full = full_scaler.transform(full_features)
    train_x, train_y = _repeat_by_branches(transformed_full, branch_labels)
    full_classifier = LogisticRegression(
        C=c,
        penalty="l2",
        solver="lbfgs",
        max_iter=2_000,
        random_state=0,
    ).fit(train_x, train_y)
    full_student = FrozenStudent(
        name="full",
        scaler=full_scaler,
        pca=None,
        classifier=full_classifier,
    )
    # Store the state transform on the full student without relying on an opaque
    # sklearn Pipeline. The two pieces are attached explicitly for serialization.
    full_student.state_scaler = state_scaler  # type: ignore[attr-defined]
    full_student.state_pca = state_pca  # type: ignore[attr-defined]

    return CandidateStudents(
        history=history_student,
        beta=beta_student,
        full=full_student,
        prior=float(branch_labels.mean()),
    )


def predict_students(
    students: CandidateStudents,
    state_graph: NDArray[np.float64],
    history: NDArray[np.float64],
    beta: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:
    state_scaler = students.full.state_scaler  # type: ignore[attr-defined]
    state_pca = students.full.state_pca  # type: ignore[attr-defined]
    state_components = state_pca.transform(state_scaler.transform(state_graph))
    full_features = np.column_stack((state_components, history))
    return {
        "prior": np.full(history.shape[0], students.prior, dtype=np.float64),
        "history": students.history.predict(history),
        "beta": students.beta.predict(beta),
        "full": students.full.predict(full_features),
    }


def _archive_scale(archive: Any, prefix: str, values: NDArray) -> NDArray[np.float64]:
    return (values - archive[f"{prefix}__scaler_mean"]) / archive[
        f"{prefix}__scaler_scale"
    ]


def _archive_pca(archive: Any, prefix: str, values: NDArray) -> NDArray[np.float64]:
    return (values - archive[f"{prefix}__pca_mean"]) @ archive[
        f"{prefix}__pca_components"
    ].T


def _archive_probability(archive: Any, prefix: str, values: NDArray) -> NDArray[np.float64]:
    logits = (
        values @ archive[f"{prefix}__classifier_coef"].T
        + archive[f"{prefix}__classifier_intercept"]
    ).reshape(-1)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -709.0, 709.0)))


def predict_frozen_archive(
    archive_path: Path | str,
    candidate: str,
    state_graph: NDArray[np.float64],
    history: NDArray[np.float64],
    beta: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:
    """Predict directly from the portable parameter archive without refitting."""

    with np.load(archive_path) as archive:
        base = f"c{candidate}"
        history_prefix = f"{base}__history"
        history_scaled = _archive_scale(archive, history_prefix, history)
        history_prediction = _archive_probability(
            archive, history_prefix, history_scaled
        )

        beta_prefix = f"{base}__beta"
        beta_scaled = _archive_scale(archive, beta_prefix, beta)
        beta_components = _archive_pca(archive, beta_prefix, beta_scaled)
        beta_prediction = _archive_probability(
            archive, beta_prefix, beta_components
        )

        state_scaled = (
            state_graph - archive[f"{base}__full_state_scaler_mean"]
        ) / archive[f"{base}__full_state_scaler_scale"]
        state_components = (
            state_scaled - archive[f"{base}__full_state_pca_mean"]
        ) @ archive[f"{base}__full_state_pca_components"].T
        full_unscaled = np.column_stack((state_components, history))
        full_prefix = f"{base}__full"
        full_scaled = _archive_scale(archive, full_prefix, full_unscaled)
        full_prediction = _archive_probability(archive, full_prefix, full_scaled)
        prior = float(archive[f"{base}__prior"][0])
    return {
        "prior": np.full(history.shape[0], prior, dtype=np.float64),
        "history": history_prediction,
        "beta": beta_prediction,
        "full": full_prediction,
    }
