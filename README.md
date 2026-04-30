# QAOA Portfolio Optimization

This repository contains Python implementations and notebooks for studying portfolio optimization with the Quantum Approximate Optimization Algorithm (QAOA).

The project includes:

- A shared classical and analytical core for portfolio-cost construction, brute-force baselines, QUBO/Ising conversion, sampling metrics, noisy sweeps, and plotting.
- A PennyLane implementation for ideal and noisy QAOA experiments.
- A Qiskit implementation for local simulation and IBM Quantum runtime workflows.
- Jupyter notebooks for exploratory runs and experiment analysis.

## Repository Layout

```text
.
|-- portfolio_qaoa_core.py
|-- IBM/
|   |-- portfolio_qaoa_qiskit.py
|   `-- ibm.ipynb
`-- pennylane/
    |-- portfolio_qaoa_pennylane.py
    `-- pennylane.ipynb
```

Generated experiment artifacts are intentionally ignored by Git, including `IBM/outputs/` and `pennylane/noisy_results/`.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Usage

Open the notebooks to reproduce or extend the experiments:

```powershell
jupyter lab
```

Or import the modules directly from Python:

```python
from portfolio_qaoa_core import get_default_worked_example, bruteforce_portfolio_baseline
from pennylane.portfolio_qaoa_pennylane import run_qaoa_experiment
from IBM.portfolio_qaoa_qiskit import run_qaoa_experiment_qiskit
```

For IBM Quantum hardware or runtime execution, configure your IBM Quantum credentials locally. Do not commit tokens or account files.

## Notes

The repository focuses on small portfolio instances where brute-force baselines are practical and useful for comparing QAOA sampling behavior.
