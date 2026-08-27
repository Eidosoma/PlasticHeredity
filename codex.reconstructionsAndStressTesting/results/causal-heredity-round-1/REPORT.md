# Causal Plastic Heredity in Cellular Automata

Profile: `reference`. State: **complete**.
Design digest: `51710c4b87bdf0f154c67741daff6b15c7ff5d97fc2a68af62fe79da354f3e8b`.

## Donor acquisition

- ECA: 272/320 donors; below-target rules `[0, 8, 150]`.
- LIFE: 384/384 donors; below-target rules `[]`.

## Registered gates

- `causal_transmission`: **False**
- `structure_matters`: **False**
- `dose_response`: **False**
- `pedigree_persistence`: **False**
- `observer_robustness`: **False**
- `environmental_memory`: **False**
- `rule_specificity`: **True**

## Primary causal contrasts

### ECA

- Structured-half minus density-random: `-0.000244140625`; 95% CI `[-0.01708984375, 0.016845703125]`.
- Structured-half minus shuffled: `-0.008056640625`; 95% CI `[-0.0244140625, 0.007568359375]`.
- Stable-maintainer structured-half minus density-random (descriptive control): `0.0`; 95% CI `[0.0, 0.0]`.
- Dose means: `[0.2880859375, 0.30908203125, 0.311767578125, 0.328857421875]`; slope CI `[0.027438964843749945, 0.07382812499999995]`.
- Pedigree half minus shuffled: `-0.03658531875925116`; 95% CI `[-0.06270733695490993, -0.010700634237142915]`.
- Native minus one-bit-neighbor transplant: `0.139404296875`; 95% CI `[0.080078125, 0.206298828125]`.
- Independent-observer directions: `{'raw4': -0.005126953125, 'multiscale': -0.006103515625}`.

### LIFE

- Structured-half minus density-random: `0.0146484375`; 95% CI `[-0.0003255208333333333, 0.029622395833333332]`.
- Structured-half minus shuffled: `0.010416666666666666`; 95% CI `[-0.00390625, 0.024739583333333332]`.
- Stable-maintainer structured-half minus density-random (descriptive control): `0.059244791666666664`; 95% CI `[0.029622395833333332, 0.0908203125]`.
- Dose means: `[0.6686197916666666, 0.6813151041666666, 0.7067057291666666, 0.7014973958333334]`; slope CI `[0.0035156249999999294, 0.09909179687499992]`.
- Pedigree half minus shuffled: `-0.11082480342825285`; 95% CI `[-0.15597380851795575, -0.06990134090711127]`.
- Native minus one-bit-neighbor transplant: `0.27619485294117646`; 95% CI `[0.22089460784313728, 0.32598422181372544]`.
- Independent-observer directions: `{'terminal2x2': 0.017578125, 'components': 0.006184895833333333}`.

## Environmental memory

- ECA: after one `0.575` (p `9.999e-05`); after eight `0.492` (p `0.7878`).
- LIFE: after one `0.645` (p `9.999e-05`); after eight `0.502` (p `0.6533`).

## Interpretation boundary

A passed causal gate means inherited structured lattice information regenerated the acquired observer-level form more reliably than registered mass- and density-matched controls. It does not establish biochemical genetics, agency, or observer-independent organismhood. All nulls, reversals, missing donors, and truncated conditions remain in the machine-readable artifacts.
