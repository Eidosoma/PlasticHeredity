#!/usr/bin/env python3
"""Rebuild frozen model archives from retained development arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from plastic_heredity.experiment import save_frozen_students, save_model_contract
from plastic_heredity.models import fit_students


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    arguments = parser.parse_args()
    manifest = json.loads((arguments.results / "manifest.json").read_text())
    settings = manifest["experiment"]
    landmarks = len(settings["development"]["landmarks"])
    arrays = np.load(arguments.results / "analysis_arrays.npz")
    state_graph = arrays["development_state_graph"]
    history = arrays["development_history"]
    beta = arrays["development_beta"]
    labels = arrays["development_targets"]
    within_matrix = np.arange(state_graph.shape[0]) % (2 * landmarks)

    students = {}
    for candidate, selected in (
        ("02", within_matrix < landmarks),
        ("03", within_matrix >= landmarks),
    ):
        students[candidate] = fit_students(
            state_graph[selected],
            history[selected],
            beta[selected],
            labels[selected],
            pca_components=settings["pca_components"],
            c=settings["logistic_c"],
        )
    save_frozen_students(arguments.results / "frozen_models.npz", students)
    save_model_contract(arguments.results / "model_contract.json")
    print(arguments.results / "frozen_models.npz")


if __name__ == "__main__":
    main()
