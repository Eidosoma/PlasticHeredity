# Causal Plastic Heredity in Cellular Automata

Profile: `pilot`. State: **complete**.
Design digest: `a3510263c3790ec15542b0bc138e463b8ec8c9f9125ba48082eb393a8871d11a`.

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
- Structured-half minus shuffled: `-0.0625`; 95% CI `[-0.14583333333333334, 0.041666666666666664]`.
- Stable-maintainer structured-half minus density-random (descriptive control): `None`; 95% CI `[None, None]`.
- Dose means: `[0.23958333333333334, 0.20833333333333334, 0.21875, 0.3125]`; slope CI `[-0.012500000000000006, 0.21249999999999994]`.
- Pedigree half minus shuffled: `0.052083333333333336`; 95% CI `[-0.021093749999999984, 0.13541666666666666]`.
- Native minus one-bit-neighbor transplant: `0.057291666666666664`; 95% CI `[-0.010416666666666666, 0.14583333333333334]`.
- Independent-observer directions: `{'raw4': -0.010416666666666666, 'multiscale': -0.010416666666666666}`.

### LIFE

- Structured-half minus density-random: `0.25`; 95% CI `[0.0625, 0.4375]`.
- Structured-half minus shuffled: `0.125`; 95% CI `[0.0, 0.25]`.
- Stable-maintainer structured-half minus density-random (descriptive control): `0.10416666666666667`; 95% CI `[0.020833333333333332, 0.19791666666666666]`.
- Dose means: `[0.125, 0.25, 0.25, 0.1875]`; slope CI `[-0.525, 0.7999999999999998]`.
- Pedigree half minus shuffled: `0.0`; 95% CI `[0.0, 0.0]`.
- Native minus one-bit-neighbor transplant: `0.2941176470588235`; 95% CI `[0.20588235294117646, 0.38235294117647056]`.
- Independent-observer directions: `{'terminal2x2': 0.25, 'components': 0.1875}`.

## Environmental memory

- ECA: after one `0.438` (p `0.8621`); after eight `0.521` (p `0.4486`).
- LIFE: after one `0.000` (p `1`); after eight `0.000` (p `1`).

## Interpretation boundary

A passed causal gate means inherited structured lattice information regenerated the acquired observer-level form more reliably than registered mass- and density-matched controls. It does not establish biochemical genetics, agency, or observer-independent organismhood. All nulls, reversals, missing donors, and truncated conditions remain in the machine-readable artifacts.
