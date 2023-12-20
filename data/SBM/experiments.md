| Experiment | $n$ | $Q$ |                                $\alpha$                               |                                     $\pi$                                    |
|:----------:|:---:|:---:|:---------------------------------------------------------------------:|:----------------------------------------------------------------------------:|
|      1     |  30 |  3  |                     $\alpha \sim \text{Dir}(1.5)$                     |                 $\pi = U U^T \times \frac{1}{2}$ (normalized)                |
|      2     | 500 |  3  |                     $\alpha \sim \text{Dir}(1.5)$                     |                 $\pi = U U^T \times \frac{1}{2}$ (normalized)                |
|      3     | 200 |  3  |                      $\alpha_i = (\frac{1}{Q})_i$                     |           $\pi_{ii} = 1-\varepsilon_1, \ \pi_{ij} = \varepsilon_2$           |
|      4     | 200 |  5  |                      $\alpha_i = (\frac{1}{Q})_i$                     |           $\pi_{ii} = 1-\varepsilon_1, \ \pi_{ij} = \varepsilon_2$           |
|      5     | 200 |  2  | $\alpha = \begin{pmatrix} \frac{9}{10} \\ \frac{1}{10} \end{pmatrix}$ | $\pi = \begin{pmatrix} \varepsilon_1 & a \\ \varepsilon_2 & b \end{pmatrix}$ |
|      6     | 200 |  3  |                      $\alpha_i = (\frac{1}{Q})_i$                     |          $\pi_{ii} = \varepsilon_2, \ \pi_{ij} = 1 - \varepsilon_1$          |

In this table, $\varepsilon_1 = 0.1$, $\varepsilon_2=0.01$, $a = 0.7$ and $b=0.8$. $\text{Dir}(\delta)$ is the Dirichlet law of parameter $\delta$. $G$ is a $Q \times Q$ random matrix with i.i.d. normally distributed coefficients.