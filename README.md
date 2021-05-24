# WIENER FILTER AND KALMAN FILTER FOR FAULTY SENSOR INTERPOLATION

Wiener filter and Kalman filter are linear Bayesian estimation methods. In this context, they are used to interpolate faulty sensor readings. The repository contains Python3 scripts for:

- Full Wiener Interpolator,
- Two-Point Wiener Interpolator,
- Causal Wiener Interpolator,
- Kalman Filter,
for faulty sensor interpolation.

## Documentation

`docs/instructions.pdf` contains the necessary instructions for the assignment. `docs/solutions.pdf` contains theory and algorithms.

## Installation

Clone this repository and install the requirments using
```shell
git clone https://github.com/kamath-abhijith/Wiener_Filter
conda create --name <env> --file requirements.txt
```

## Run

- Run `interp_wiener.py` to generate Figure 1.
- Run `interp_kalman.py` to generate Figure 2 and 3.