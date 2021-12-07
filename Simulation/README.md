## Proof
Given a path, if it contains **even** negative edges, we then refer to it as a positive path; Otherwise, we refer to it as a negative path.

Let <img src="https://latex.codecogs.com/svg.image?p_1" title="p_1" /> denote the probabilities of starting from <img src="https://latex.codecogs.com/svg.image?V_i" title="V_i" /> and reaching at <img src="https://latex.codecogs.com/svg.image?V_j" title="V_j" /> along h-step positive paths.

Let <img src="https://latex.codecogs.com/svg.image?p_2" title="p_2" /> denote the probabilities of starting from <img src="https://latex.codecogs.com/svg.image?V_i" title="V_i" /> and reaching at <img src="https://latex.codecogs.com/svg.image?V_j" title="V_j" /> along h-step negative paths.

We have that the probability of traversing from <img src="https://latex.codecogs.com/svg.image?V_i" title="V_i" /> to <img src="https://latex.codecogs.com/svg.image?V_j" title="V_j" /> along h-step paths is <img src="https://latex.codecogs.com/svg.image?\hat{P}^h_{ij}" title="\hat{P}^h_{ij}" />, i.e., <img src="https://latex.codecogs.com/svg.image?p_1+p_2" title="p_1+p_2" />.
And the difference between <img src="https://latex.codecogs.com/svg.image?p_1" title="p_1" /> and <img src="https://latex.codecogs.com/svg.image?p_2" title="p_2" /> is given by <img src="https://latex.codecogs.com/svg.image?p_1-p_2=P^h_{ij}" title="p_1-p_2=P^h_{ij}" />.  

We then have <img src="https://latex.codecogs.com/svg.image?p_1=\frac{((p_1+p_2)+(p_1-p_2))}{2}=\frac{(\hat{P}^h_{ij}+P^h_{ij})}{2}" title="p_1=\frac{((p_1+p_2)+(p_1-p_2))}{2}=\frac{(\hat{P}^h_{ij}+P^h_{ij})}{2}" />.
According to the law of large numbers, we have <img src="https://latex.codecogs.com/svg.image?p_1=\frac{\lim_{W->\infty}\sharp_h(V_i,V_j)}{W}" title="p_1=\frac{\lim_{W->\infty}\sharp_h(V_i,V_j)}{W}" />

That is, we have <img src="https://latex.codecogs.com/svg.image?\frac{\lim_{W->\infty}\sharp_h(vi,vj)}{W}=\frac{(\hat{P}^h_{ij}+P^h_{ij})}{2}" title="\frac{\lim_{W->\infty}\sharp_h(vi,vj)}{W}=\frac{(\hat{P}^h_{ij}+P^h_{ij})}{2}" />.

## Instructions for simulation results
Furthermore, we provide the code of simulation experiments to empirically validate Eq.(6).

Of course, W cannot be infinite when using explicit sampling.
Thus, we first set W (i.e., `num_walks` in the ``exp.py) to a relatively large value (e.g., 10,000).
We then perform random walk and count the number of samples about positve proximity. Next, we can calculate the left term and right term of Eq.(6).

For example, in the following graph, when setting W=10,000 and h=2, we have

<img src="test.png" width="40%" height="40%">

<img src="https://latex.codecogs.com/svg.image?\frac{\sharp_h(V_i,V_j)}{W}" title="\frac{\sharp_h(V_i,V_j)}{W}" /> = \[0.5071 0.     0.2448 0.     0.     0.    \]

<img src="https://latex.codecogs.com/svg.image?\frac{(\hat{\mathbf{P}}^h)_{ij}&plus;(\mathbf{P}^h)_{ij}}{2}" title="\frac{(\hat{\mathbf{P}}^h)_{ij}+(\mathbf{P}^h)_{ij}}{2}" /> = [0.5  0.   0.25 0.   0.   0.  ]


<img src="https://latex.codecogs.com/svg.image?\frac{\sharp_h(V_i,V_j)}{W}" title="\frac{\sharp_h(V_i,V_j)}{W}" /> can approximate <img src="https://latex.codecogs.com/svg.image?\frac{(\hat{\mathbf{P}}^h)_{ij}&plus;(\mathbf{P}^h)_{ij}}{2}" title="\frac{(\hat{\mathbf{P}}^h)_{ij}+(\mathbf{P}^h)_{ij}}{2}" /> with lower error as W increases.
