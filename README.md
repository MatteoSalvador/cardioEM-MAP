# Fast and robust parameter estimation with uncertainty quantification for the cardiac function

This repository contains the code accompanying the paper [1]. We employ an artificial neural network-based reduced-order model for cardiac electromechanics coupled with a physics-based 0D closed-loop blood circulation model [2] to perform fast parameter identification with inverse uncertainty quantification via Maximum a Posteriori estimation and Hamiltonian Monte Carlo by using a single core standard computer.

## Installation

1. Install a conda environment containing all the required packages:

```bash
conda create -n envcardioEM-MAP python=3.7.11 numpy=1.21.5 matplotlib=3.5.1 pandas=1.3.4 scipy=1.7.3 mpi4py=3.0.3
conda activate envcardioEM-MAP
conda install -c anaconda scikit-learn
pip install --upgrade "jax[cpu]"
```

2. Clone this repository by typing:

```bash
git clone https://github.com/MatteoSalvador/cardioEM-MAP.git
```

3. Remember to activate the conda environment `envcardioEM-MAP` by typing `conda activate envcardioEM-MAP` (in case it is not already active from the installation procedure at point 1).

4. Choose the test case (`'LV'`, `'atria'` `'all'`) to run in `run_MAP_estimation_ANN.py`, along with the amount of noise in the observations (`noise_std`) and the number of trials (`n_trials`).

5. Execute the Python script `run_MAP_estimation_ANN.py`.

Note that also forward numerical simulations can be performed by using the Python script `run_circulation_ANN.py`.

## Authors (alphabetical order)

- Francesco Regazzoni (<francesco.regazzoni@polimi.it>)
- Matteo Salvador (<matteo1.salvador@polimi.it>)

## References

[1] M. Salvador, F. Regazzoni, L. Dede', A. Quarteroni. [Fast and robust parameter estimation with uncertainty quantification for the cardiac function](https://www.sciencedirect.com/science/article/pii/S016926072300069X). *Computer Methods and Programs in Biomedicine* (2023).

[2] F. Regazzoni, M. Salvador, L. Dede', A. Quarteroni. [A Machine Learning method for real-time numerical simulations of cardiac electromechanics](https://www.sciencedirect.com/science/article/pii/S004578252200144X). *Computer Methods in Applied Mechanics and Engineering* (2022).
