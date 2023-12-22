# A mixture model for random graphs

Project for the course "Probabilistic Graphical Models" of the MVA master at ENS Paris-Saclay.

Paper : [A mixture model for random graphs](https://inria.hal.science/inria-00070186/document), Jean-Jacques Daudin, Franck Picard, Stéphane Robin, 2006.

This project implements in python the variational EM algorithm proposed by Daudin et. al. to estimate the parameters of a Stochastic Block Model.

## Report

The report is available [here](report/report.pdf).

## Code

The main code is available in the `src` folder. It contains an implementation of the variational EM algorithm proposed by Daudin et. al. to estimate the parameters of a Stochastic Block Model.

## Notebooks

The following notebooks were left in the repository for the sake of completeness, but they are not necessary to run the code.

| Name | Content |
| --- | --- |
| `experiments_real.ipynb` | Experiments on the Cora and Karate Club datasets. |
| `experiments_SBM.ipynb` | Experiments on a dataset synthetically generated with the SBM model. |
| `fixed_point_convergence.ipynb` | An investigation into the convergence of the fixed point algorithm proposed by Daudin et. al. |
| `implementations_speed.ipynb` | A comparison of the speeds of different implementations of the algorithm. |
| `experiments_newman.ipynb` | Experiments with the Newman variant of the algorithm. |
| `metrics.ipynb` | Test some clustering metrics. |

## Contributors

[@bastienlc](https://github.com/bastienlc),
[@TheiloT](https://github.com/TheiloT),
[@s89ne](https://github.com/s89ne)
