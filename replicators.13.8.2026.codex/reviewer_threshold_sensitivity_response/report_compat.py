"""Run frozen reporting with a pandas/Matplotlib compatibility adapter.

The scientific runner and scoring module are protocol-hashed and remain
unchanged.  Matplotlib 3.5's private ``_check_1d`` helper attempts deprecated
``Series[:, None]`` indexing under pandas 2.x; converting Series inputs to
NumPy restores the behavior expected by that Matplotlib release.
"""

from __future__ import annotations

import matplotlib.axes._base as axes_base
import pandas as pd

from reviewer_threshold_sensitivity_response import run_sensitivity


_ORIGINAL_CHECK_1D = axes_base._check_1d


def _pandas_compatible_check_1d(value: object) -> object:
    if isinstance(value, pd.Series):
        value = value.to_numpy()
    return _ORIGINAL_CHECK_1D(value)


def main() -> None:
    axes_base._check_1d = _pandas_compatible_check_1d
    run_sensitivity.report()


if __name__ == "__main__":
    main()
