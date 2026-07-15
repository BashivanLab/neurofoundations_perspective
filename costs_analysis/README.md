# Cost model for neuro-foundation model data strategies

Reproduces Box 7 in the manuscript: estimated data costs (acquisition,
curation/generation, and storage) for four classes of neuro-foundation
model — **Exo-X-simulation**, **Exo-X-public**, **Exo-N/Endo-X**, and
**Endo-N** — as a function of the number of data points (10^6–10^10)
and project duration (1–120 months). See Supplementary Appendix A of the
manuscript for the full derivation and parameter assumptions; the
markdown cell at the top of the notebook also summarizes them.

## System requirements

- Python 3.9–3.12
- Packages: `numpy`, `matplotlib` (both standard; any recent version works)
- No GPU or non-standard hardware required — all cost functions are
  closed-form (no simulation, training, or external data access)
- Tested on: macOS and Linux, Python 3.10

## Installation guide

```bash
pip install numpy matplotlib jupyter
```

**Typical install time:** under 2 minutes on a normal desktop computer
(both packages are common and typically already cached/available).

## Demo / instructions for use

```bash
jupyter notebook cost_calculations.ipynb
```

Run all cells top to bottom. The notebook is fully self-contained — all
parameters (labor rates, hardware costs, per-data-point storage sizes,
etc.) are defined in the second code cell, and every figure is generated
from those parameters plus the two grids defined in the "Grid + plot"
section (`n_points_grid = np.logspace(6, 10, 36)`,
`months_grid = np.linspace(1, 120, 36)`). No external dataset or network
access is required.

**Expected output** (saved to the notebook's working directory):

| File | Description | Box 7 panel |
|---|---|---|
| `cost_vs_data_points_and_time_surface.pdf` | 3D cost surface vs. data points and time, all 4 classes | a |
| `cost_breakdown_pie_charts.pdf` | Acquisition/curation/storage split per class at N=10^9, 120 months | b |
| `cost_vs_data_points.pdf` | Cost vs. number of data points at 120 months | c |
| `total_cost_vs_time.pdf` | Cost vs. project duration at N=10^9 data points | d |

**Expected run time for demo:** under 1 minute on a normal desktop
computer — the most expensive step evaluates the cost functions over a
36×36 grid, which is a closed-form calculation with no iterative solving.

## Reproduction instructions

To explore alternative assumptions, edit the `params`, `hardware`, or
`per_diem` dictionaries in the second code cell (e.g. different labor
rates, hardware costs, or storage prices per Supplementary Tables
S1–S4) and re-run all cells to regenerate the four figures with the new
assumptions.
