# Minimal work protocols for inertial particles in nonharmonic traps

Accompanying code for [Minimal work protocols for inertial particles in nonharmonic traps](https://doi.org/10.1103/PhysRevE.111.034127)

This package computes numerically optimal (minimal-work) driving protocols for a Brownian particle in a nonharmonic trap, using perturbative corrections for underdamped (inertial) dynamics parameterised by a small inertia parameter `ε`.

A user guide of the procedure can be found in the [`Minimal_Work_Protocols.ipynb`](Minimal_Work_Protocols.ipynb). 

---
 
## Installation
 
Clone the repository and install dependencies:
 
```bash
git clone https://github.com/julia-sand/entropy_production.git
cd entropy_production
pip install -r requirements.txt
pip install -e .
```
 
### Requirements
 
| Package | Version |
|---|---|
| POT | 0.9.5 |
| numpy | 1.26.4 |
| scipy | 1.13.0 |
| scikit-learn | 1.4.2 |
| matplotlib | 3.8.4 |
| pandas | 2.2.2 |
 
---

To compute the overdamped problem (an optimal transport problem)
```
run python entropy_production/sinkhorn.py --args**
```

## Usage
 
### Step 1: Solve the overdamped problem
 
```bash
python src/ep/sinkhorn.py --help   # list all available options
python src/ep/sinkhorn.py --args
```
 
**Output:** a CSV file containing the overdamped results, and a text file recording the run parameters.
 
### Step 2: Compute underdamped corrections
 
```bash
python src/ep/perturbation/distributionanddrift.py --args
```
 
This reads the overdamped CSV produced in Step 1 (via `ep.utils.datafetch`) and appends new columns for the underdamped distribution and drift. Output behaviour can be customised at the bottom of `distributionanddrift.py`.
 
### Step 3: Visualise results
 
| Plot | Script |
|---|---|
| Cumulants | `cumulantsplot.py` |
| Drift and distribution | `distributionanddrift.py` |
| Joint distribution (Girsanov) | `girsanovjoint.py` then `girsanovplot.py` |
| Final distribution histograms | `histograms.py` |
| Momentum variance vs `g` | `momvarplot.py` |
| Entropy production vs time | `totalcosts.py` then `totalcostplot.py` |
 
---
 
## Repository Structure
 
```
entropy_production/
├── src/ep/
│   ├── perturbation/
│   │   └── functions.py          # Perturbation class: distributions, drifts, cumulants
│   └── utils/
│       ├── datafetch.py          # CSV loading utilities
│       ├── parser.py             # Command-line argument parsing
│       └── misc.py               # Parameter file loading
├── Minimal_Work_Protocols.ipynb  # End-to-end worked example
├── requirements.txt
├── LICENSE.txt                   # GPL-3.0
└── README.md
```
---
 
## Citation
 
If you use this code, please cite:
 
```bibtex
@article{PhysRevE.111.034127,
  title   = {Minimal work protocols for inertial particles in nonharmonic traps},
  journal = {Physical Review E},
  volume  = {111},
  pages   = {034127},
  year    = {2025},
  doi     = {10.1103/PhysRevE.111.034127}
}
```
 
---
 
## License
 
[GPL-3.0](LICENSE.txt)
 

