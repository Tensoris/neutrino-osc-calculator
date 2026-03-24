# neutrino-osc-calculator

**A Lightning-Fast Three-Flavor Neutrino Oscillation Calculator in Constant-Density Matter with Built-In Uncertainty Propagation**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Authors:** Aaryan Chaulagain, Anju Dhakal  
**Affiliation:** Independent Researcher, Kathmandu, Nepal  
**Contact:** aaryan1379@xavier.edu.np · anju1271@xavier.edu.np  
**Version:** v1.0 | August 2025

---

## Overview

This repository provides a Python implementation of three-flavor neutrino oscillation
probabilities in constant-density matter, combining:

- **Exact Hamiltonian diagonalization** (Algorithm 1) — unitarity preserved to < 10⁻¹⁰
- **Compact O(α²) perturbative approximation** (Algorithm 2) — ~27× faster than exact
- **Hybrid solver** (Algorithm 5) — automatically switches to exact near MSW resonance
- **Monte Carlo uncertainty propagation** (Algorithm 3) — NuFIT 6.0 parameter covariance
- **Jacobian linearization** (Algorithm 4) — instant confidence bands

All NuFIT 6.0 parameters and benchmarks match those reported in the accompanying paper.

---

## Installation

```bash
git clone https://github.com/YourUsername/neutrino-osc-calculator.git
cd neutrino-osc-calculator
pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.8, NumPy ≥ 1.24, Matplotlib ≥ 3.7

No external neutrino libraries are required.

---

## Quick Start

```python
from utils.constants import get_best_fit, DEFAULT_BASELINE, DEFAULT_DENSITY, DEFAULT_YE
from solvers.engines import exact_solver, perturbative_solver, hybrid_solver
import numpy as np

E_grid = np.linspace(0.3, 5.0, 500)   # GeV
params = get_best_fit()                 # NuFIT 6.0 normal ordering

P_mue, P_mumu = exact_solver(E_grid, DEFAULT_BASELINE, DEFAULT_DENSITY, DEFAULT_YE, params)
print(f"Peak appearance probability: {P_mue.max():.4f}")
```

---

## Reproducing Paper Figures

```bash
python main.py
```

This generates all five publication figures: `appearance_plot.png`, `disappearance_plot.png`,
`comparison_plot.png`, `error_plot.png`, `performance_plot.png`.

---

## Repository Structure

```
neutrino-osc-calculator/
│
├── README.md                     ← This file
├── LICENSE                       ← CC BY 4.0
├── requirements.txt              ← Python dependencies
├── setup.py                      ← Package installation
├── .zenodo.json                  ← Zenodo metadata (auto-read on upload)
│
├── main.py                       ← Reproduces all paper figures
│
├── core/
│   ├── __init__.py
│   └── physics.py                ← PMNS matrix, matter potential
│
├── solvers/
│   ├── __init__.py
│   └── engines.py                ← exact_solver, perturbative_solver, hybrid_solver
│
├── uncertainty/
│   ├── __init__.py
│   └── error_prop.py             ← monte_carlo_propagation, jacobian_propagation
│
├── utils/
│   ├── __init__.py
│   └── constants.py              ← NuFIT 6.0 parameters, physical constants
│
├── paper/
│   ├── main.tex                  ← Full LaTeX manuscript
│   └── figures/                  ← Generated figure PNGs (gitignored if large)
│       ├── appearance_plot.png
│       ├── disappearance_plot.png
│       ├── comparison_plot.png
│       ├── error_plot.png
│       └── performance_plot.png
│
└── tests/
    ├── __init__.py
    └── test_unitarity.py         ← Unitarity and vacuum-limit checks
```

---

## Citation

If you use this code, please cite the accompanying paper and this software release:

```bibtex
@article{chaulagain2026neutrino,
  title   = {A Lightning-Fast Three-Flavor Neutrino Oscillation Calculator
             in Constant-Density Matter with Built-In Uncertainty Propagation},
  author  = {Chaulagain, Aaryan and Dhakal, Anju},
  journal = {[venue]},
  year    = {2026},
  doi     = {10.5281/zenodo.XXXXXXX}
}

@software{chaulagain2026code,
  author    = {Chaulagain, Aaryan and Dhakal, Anju},
  title     = {neutrino-osc-calculator v1.0},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.XXXXXXX},
  url       = {https://doi.org/10.5281/zenodo.XXXXXXX}
}
```

---

## License

This work is licensed under the
[Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
You are free to share and adapt this material for any purpose, provided appropriate
credit is given.

---

## Acknowledgment

The authors thank Professor Takaaki Kajita for his feedback on the theoretical
framework and his guidance on neutrino oscillation physics. Physical constants and
mixing parameters follow NuFIT 6.0 (Esteban et al., JHEP 12, 216, 2024).
