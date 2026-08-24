"""Exhaustive one-molecule interventions directed by local Phi-r."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray

from .gard import InterventionDecision
from .config import CausalConfig
from .information import CausalTrajectory, fit_causal_trajectory


IntArray = NDArray[np.int64]


@dataclass
class PhiDirectedIntervention:
    """Choose the feasible +/-1 molecule action with the extremal local score.

    The reference trajectory is fitted once on the matched untreated run. This
    is an explicit reconstruction choice because the preprint does not state
    how a time-series information measure was evaluated online.
    """

    reference: CausalTrajectory
    direction: Literal["max", "min"]
    max_size: int

    def __post_init__(self) -> None:
        if self.direction not in {"max", "min"}:
            raise ValueError("direction must be 'max' or 'min'")

    def __call__(
        self,
        parent: IntArray,
        daughter: IntArray,
        history: IntArray,
        generation: int,
    ) -> InterventionDecision:
        del history, generation
        candidates = []
        species = []
        deltas = []
        if daughter.sum() < self.max_size:
            for index in range(daughter.size):
                candidate = daughter.copy()
                candidate[index] += 1
                candidates.append(candidate)
                species.append(index)
                deltas.append(1)
        if daughter.sum() > 1:
            for index in np.flatnonzero(daughter):
                candidate = daughter.copy()
                candidate[index] -= 1
                candidates.append(candidate)
                species.append(int(index))
                deltas.append(-1)
        if not candidates:
            return InterventionDecision()
        candidate_matrix = np.asarray(candidates, dtype=np.int64)
        scores = self.reference.score_count_transitions(parent, candidate_matrix)
        selected = int(np.nanargmax(scores) if self.direction == "max" else np.nanargmin(scores))
        return InterventionDecision(
            species=species[selected],
            delta=deltas[selected],
            score=float(scores[selected]),
        )


@dataclass
class OnlinePhiDirectedIntervention:
    """Choose an extremal action using only observations available so far.

    By default the partition and local Gaussian information model are fitted
    at the first scorable fission and then held fixed, so all later candidate
    scores share a common, pre-intervention reference distribution. Optional
    per-generation refitting uses the treatment history available at that
    point. Both avoid future-data leakage. The preprint does not specify its
    online estimator, so the matched-control implementation remains available
    as a sensitivity option.
    """

    config: CausalConfig
    direction: Literal["max", "min"]
    max_size: int
    refit_each_generation: bool = False
    _reference: Optional[CausalTrajectory] = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.direction not in {"max", "min"}:
            raise ValueError("direction must be 'max' or 'min'")
        self.config.validate()

    def __call__(
        self,
        parent: IntArray,
        daughter: IntArray,
        history: IntArray,
        generation: int,
    ) -> InterventionDecision:
        del generation
        # A multivariate lagged Gaussian model requires at least six states.
        # An unusually large first leap can hit fission before that much
        # history exists; in that case the only leakage-free action is to
        # defer the intervention for this generation.
        if history.shape[0] <= self.config.lag + 4:
            return InterventionDecision()
        if self._reference is None or self.refit_each_generation:
            self._reference = fit_causal_trajectory(history, self.config)
        return PhiDirectedIntervention(
            reference=self._reference,
            direction=self.direction,
            max_size=self.max_size,
        )(parent, daughter, history, 0)
