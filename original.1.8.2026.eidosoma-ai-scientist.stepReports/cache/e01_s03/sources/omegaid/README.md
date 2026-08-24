# ΩID

[![PyPI version](https://badge.fury.io/py/omegaid.svg)](https://badge.fury.io/py/omegaid)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/dmf-archive/OmegaID)

ΩID is a Python package for calculating the integrated information decomposition (ΦID) of time series data. It is designed for high-performance computing, with optional GPU acceleration via CuPy.

## Features

- **Backend Agnostic**: Seamlessly switch between CPU (NumPy) and GPU (CuPy) for computation.
- **High Performance**: Vectorized operations and Numba-optimized functions for significant speedups.
- **Numerical Integrity**: Results are numerically consistent with the original `phyid` implementation within single/double precision tolerances.
- **Multivariate Decomposition**: Supports decomposition of systems with multiple source and target variables using the Doublet Lattice approximation.

## Installation

ΩID is available on PyPI. You can install it with `pip` or `uv pip`.

### Standard Installation (CPU only)

```bash
pip install omegaid
```

### With GPU support

To install ΩID with GPU support, you need to have a CUDA-enabled GPU and the CUDA toolkit installed. Choose the command that matches your CUDA version.

**For CUDA 12.x:**

```bash
pip install "omegaid[cuda-12x]"
```

**For CUDA 11.x:**

```bash
pip install "omegaid[cuda-11x]"
```

## Usage

### Selecting the Backend

You can select the computation backend by setting the `OMEGAID_BACKEND` environment variable before running your Python script.

- **For NumPy (default):**

    ```bash
    export OMEGAID_BACKEND=numpy
    ```

- **For CuPy:**

    ```bash
    export OMEGAID_BACKEND=cupy
    ```

If the variable is not set, OmegaID will default to using NumPy.

### Example

Here is a simple example of how to use `omegaid` to calculate the Phi-ID decomposition for a multivariate system.

```python
import numpy as np
from omegaid.core.decomposition import calc_phiid_multivariate
from omegaid.utils.backend import set_backend

# For programmatic control, you can also use set_backend
# set_backend('cupy') 

# Generate some random time series data (4 sources, 2 targets)
n_sources = 4
n_targets = 2
n_samples = 10000
tau = 1
sources = np.random.randn(n_sources, n_samples)
targets = np.random.randn(n_targets, n_samples)

# Calculate Phi-ID using the Doublet Lattice approximation
atoms_res, _ = calc_phiid_multivariate(sources, targets, tau)

# Print a synergistic atom, e.g., between source 0 and target 1
print("Synergy (s0, t1):", atoms_res.get((0, 1), "N/A"))
```

## Performance

The package has been benchmarked for various systems. The results below show the performance against the original `phyid` (where applicable) and between CPU/GPU backends.

| Test Case               | Implementation          | Time (s)      | Speedup         |
| :---------------------- | :---------------------- | :------------ | :-------------- |
| 2x2 CCS (100k)          | omegaid_2x2_cpu         | 0.1390        | 1.18x           |
|                         | omegaid_2x2_gpu         | 0.0947        | 1.47x           |
|                         | phyid                   | 0.1640        | 1.00x           |
| 2x2 MMI (100k)          | omegaid_2x2_cpu         | 0.1168        | 1.07x           |
|                         | omegaid_2x2_gpu         | 0.4511        | 0.26x           |
|                         | phyid                   | 0.1247        | 1.00x           |
| 4x2 CCS (100k)          | omegaid_4x2_cpu         | 0.3962        | 1.00x           |
|                         | omegaid_4x2_gpu         | 0.1892        | 2.09x           |
| 4x2 MMI (100k)          | omegaid_4x2_cpu         | 0.4043        | 1.00x           |
|                         | omegaid_4x2_gpu         | 0.2558        | 1.58x           |
| 4x4 MMI (10k)           | omegaid_4x4_cpu         | 1.8373        | 1.00x           |
|                         | omegaid_4x4_gpu         | 0.8201        | 2.24x           |

The results demonstrate that for more complex multivariate systems (e.g., 4x2 and 4x4), the CuPy backend provides a consistent and significant performance advantage due to the fully vectorized implementation. For simpler 2x2 systems, the Numba-optimized NumPy backend remains highly competitive.

## License

This project is licensed under the BSD 3-Clause License.
