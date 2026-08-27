# Causal Plastic Heredity in Cellular Automata

Profile: `pilot`. State: **complete**.
Design digest: `f5e565730254e5ea5459704f7cf5c99dcb41e01c4548d45dffbf31d2e4c0aa50`.

## Donor acquisition

- ECA: 24/32 donors; below-target rules `[0, 8]`.
- LIFE: 28/32 donors; below-target rules `[119272, 125408]`.

## Registered gates

- `causal_transmission`: **False**
- `structure_matters`: **False**
- `dose_response`: **False**
- `pedigree_persistence`: **False**
- `observer_robustness`: **False**
- `environmental_memory`: **False**
- `rule_specificity`: **False**

## Primary causal contrasts

### ECA

- Structured-half minus density-random: `-0.03125`; 95% CI `[-0.125, 0.0625]`.
- Structured-half minus shuffled: `-0.03125`; 95% CI `[-0.14583333333333334, 0.08333333333333333]`.
- Dose means: `[0.23958333333333334, 0.20833333333333334, 0.21875, 0.3125]`; slope CI `[-0.012500000000000006, 0.21249999999999994]`.
- Pedigree half minus shuffled: `0.052083333333333336`; 95% CI `[-0.021093749999999984, 0.13541666666666666]`.
- Native minus one-bit-neighbor transplant: `0.057291666666666664`; 95% CI `[-0.010416666666666666, 0.14583333333333334]`.
- Independent-observer directions: `{'raw4': -0.010416666666666666, 'multiscale': -0.010416666666666666}`.

### LIFE

- Structured-half minus density-random: `0.125`; 95% CI `[0.044642857142857144, 0.20558035714285694]`.
- Structured-half minus shuffled: `0.0625`; 95% CI `[-0.017857142857142856, 0.14285714285714285]`.
- Dose means: `[0.26785714285714285, 0.4107142857142857, 0.5446428571428571, 0.5357142857142857]`; slope CI `[0.1963392857142857, 0.5714285714285714]`.
- Pedigree half minus shuffled: `0.11041666666666666`; 95% CI `[0.03958333333333333, 0.196875]`.
- Native minus one-bit-neighbor transplant: `0.2941176470588235`; 95% CI `[0.20588235294117646, 0.38235294117647056]`.
- Independent-observer directions: `{'terminal2x2': 0.08928571428571429, 'components': 0.03571428571428571}`.

## Environmental memory

- ECA: after one `0.438` (p `0.6663`); after eight `0.521` (p `0.2108`).
- LIFE: after one `0.469` (p `0.4715`); after eight `0.438` (p `0.6444`).

## Interpretation boundary

A passed causal gate means inherited structured lattice information regenerated the acquired observer-level form more reliably than registered mass- and density-matched controls. It does not establish biochemical genetics, agency, or observer-independent organismhood. All nulls, reversals, missing donors, and truncated conditions remain in the machine-readable artifacts.
