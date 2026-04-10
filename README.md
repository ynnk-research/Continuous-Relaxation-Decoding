# Continuous Relaxation Decoding for CSS Quantum LDPC Codes
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19484007.svg)](https://doi.org/10.5281/zenodo.19484007) 
[![Code License: Apache 2.0](https://img.shields.io/badge/Code_License-Apache_2.0-blue.svg)](LICENSE)
[![Doc License: CC BY 4.0](https://img.shields.io/badge/Doc_License-CC_BY_4.0-green.svg)](LICENSE-CC-BY.txt)


**Paper:** *Continuous Relaxation Decoding for CSS Quantum LDPC Codes: Energy Landscape, Basin Geometry, Gradient Dynamics, and Hardware Implementation*  
**Author:** Yannick Schmitt  
**Date:** April 2026  
**DOI:** [10.5281/zenodo.19484007](https://doi.org/10.5281/zenodo.19484007)  
**Status:** Preprint 1.0.0


## What this repository contains

```
paper/
  Continuous_Relaxation_Decoding_for_CSS_Quantum_LDPC_Codes.tex   # full source
  Continuous_Relaxation_Decoding_for_CSS_Quantum_LDPC_Codes.pdf   # compiled paper

script/
  continuous_relaxation_decoder.py   # core library (decoder, code builders, GF(2) tools)
  verification_ContinuousRelaxationDecoder.py   # unified verification script

misc_computation/
  o1_convergence_radius.py   # basin geometry: three nested radii
  o2_momentum_escape.py        # momentum decoder comparison (GD / HB / NAG / Langevin)
  o3_decoder_threshold.py      # DSP measurement, failure mode census
  o4_css_generalisation.py     # λ* universality across CSS code families
  o5_hardware_acceleration.py  # FLOP model, serial/batch/PyTorch throughput
  cd_smart_explorer.py   # causal diamond seed search, 64 augmenting rows, B4 orbits
  cd_smart_explorer_augmentation   # level-2 augmentation and [[468,36]] ISD certification
```

---

## The decoder

Binary error variables $e_q \in \{0,1\}$ are lifted to continuous spin variables $v_q \in \mathbb{R}$. A squared-spring energy

$$H(v) = \sum_q \frac{\lambda}{4}(v_q^2-1)^2 + \sum_j \frac{(1-P_j)^2}{4}, \quad P_j = \prod_{q \in \text{supp}(j)} v_q$$

has global minima exactly at valid codewords ($H = 0$, $\nabla H = 0$). Gradient descent from a syndrome-consistent initialisation converges to the nearest codeword. The energy is **syndrome-agnostic** — the decoder is a codeword projector, not a syndrome solver.


## Key results

| Result | Value |
|---|---|
| Primary testbed | [[193, 25, d=4]] D4 causal diamond HGP code |
| Secondary testbeds | [[112, 4, (6,6)]] and [[176, 32, (3,6)]] augmented-seed codes |
| Provably perfect decoding | t ≤ 1 (monotone path theorem); t = 2 empirically |
| Pseudo-codeword barrier heights | ΔE ∈ [1.0, 4.0] at λ = 0.5 |
| Optimal λ heuristic | λ\* ≈ (3/16)d̄, confirmed on 6 of 8 code families |
| Girth–trap hypothesis | **Falsified**: girth-8 code traps at 2.5× the rate of girth-4 |
| GPU throughput | 86 535 syn/s at batch 16 384 (~28 000× over serial Python) |
| FLOP vs MWPM | 8× fewer than O(N³) bound; 18× more than sparse-Blossom |


## Analytical results verified to machine precision

All three theorems hold for any CSS code and are verified numerically across eight independent code families.

**Theorem 3.4 — Critical points**  
The linear spring energy has no fixed points at valid codewords. The squared spring achieves $H(v^\*) = 0$ and $\nabla H(v^\*) = \mathbf{0}$ exactly.

**Theorem 4.1 — Single-error gradient**  
For a single-qubit error at $q^\*$: $\partial H / \partial v\_{q^\*} = -\deg\_Z(q^\*)$ (restoring force), and $\partial H / \partial v\_q = +A\_{q,q^\*}$ for $q \neq q^\*$ (positive, no factor of 1/2).

**Theorem 4.3 — Exact Hessian at valid codewords**  
$\mathbf{H} = 2\lambda I + \frac{1}{2}A$ where $A = H\_Z^T H\_Z$. Diagonal: $\mathbf{H}\_{qq} = 2\lambda + \deg\_Z(q)/2$. Off-diagonal: $\mathbf{H}\_{qr} = A\_{qr}/2$.


## Repository layout

**To reproduce the paper's tables and figures** → run `verification_ContinuousRelaxationDecoder.py`. It is self-contained (no external dependencies beyond NumPy/SciPy), covers every theorem, proposition, and observation, and prints `PASS / FAIL` for each claim.

**To understand the basin geometry in depth** → `o1_convergence_radius_2.py` computes the three nested radii ($r_\text{loc}$, $r_\text{barrier}$, $r_\text{emp}$) and their $\lambda$-scaling.

**To reproduce the momentum decoder comparison** → `o2_momentum_escape.py` runs all four optimiser variants, the γ-sweep, the Langevin T₀-sweep, and the restart-interval sensitivity test.

**To reproduce the threshold and failure-mode analysis** → `o3_decoder_threshold.py` builds the logical basis, runs the $P_L(p)$ curve, and classifies syndrome vs logical failures.

**To reproduce the λ\* universality result across code families** → `o4_css_generalisation.py` constructs the code zoo and sweeps λ on each.

**To reproduce the hardware benchmarks** → `o5_hardware_acceleration.py` measures serial, batch-NumPy, and PyTorch throughput and compares to the FLOP model.

**To reproduce the causal diamond geometry analysis** (64 augmenting rows, B4 orbits, [[208,16,6]] distance certification) → `cd_smart_explorer_fixed.py` and `cd_smart_explorer_fixed_addition.py`.


## Dependencies

```
numpy
scipy          # o1 only (scipy.linalg.eigh)
torch          # o5 only, optional; falls back to CPU if CUDA unavailable
```

All misc_computation scripts require `continuous_relaxation_decoder.py` on the Python path. The simplest approach:

```bash
# from the repo root
PYTHONPATH=script python3 script/verification_ContinuousRelaxationDecoder.py
PYTHONPATH=script python3 misc_computation/o1_convergence_radius_2.py
```

The `misc_computation` scripts do `sys.path.insert(0, '/home/crd')` by default — change that path to point to the `script/` directory.


## Running the verification script

```bash
# Full run (~30–60 min depending on hardware)
python3 verification_ContinuousRelaxationDecoder.py

# Quick smoke-test (~5 min): set FAST=True at the top of the script
```


## Citation
If you use this work, please cite it as:

```bibtex
@misc{schmitt2026crd,
  author    = {Yannick Schmitt},
  title     = {Continuous Relaxation Decoding for {CSS} Quantum {LDPC} Codes:
               Energy Landscape, Basin Geometry, Gradient Dynamics, and
               Hardware Implementation},
  year      = {2026},
  doi       = {10.5281/zenodo.19484007},
  url       = {https://doi.org/10.5281/zenodo.19484007}
}
```

> Yannick Schmitt. (2026). Continuous Relaxation Decoding for CSS Quantum LDPC Codes. Zenodo. https://doi.org/10.5281/zenodo.19484007


## License
 * The source code in this repository is licensed under the [Apache License 2.0](LICENSE).
 * The documentation, LaTeX source files, and PDF papers are licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](LICENSE-CC-BY.txt).
