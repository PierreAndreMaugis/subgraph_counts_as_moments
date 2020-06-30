# Moments vs Cumulants
## The context
Given two graph samples (two lists of graphs all of the same order) we want to test for the null of both being drawn
    iid from the same distribution. To achieve this test we can either use the graph
    [moments](https://arxiv.org/abs/1701.00505) or the graph [cumulants](https://arxiv.org/abs/2002.03959).
    
Under the null, and after standardization, both approaches yield chi2 limit distributions. We want to know if the
    moment based test is more powerful than the cumulant based one. Therefore, we set a list of graphs, produce the
    cumulants associated to these graphs, and collate the moments that need to be computed to estimate the said
    cumulants. We thus get two tests based on the same information and converging to the same limit distribution.
    
To compare the tests we use Pitman asymptotic efficiency for a range of parametric nulls and alternatives. The parametric
    model we consider is that of [Example 5.1.3](https://arxiv.org/pdf/1808.04855.pdf): a 2 block block model, of rank
    1, which we constrained to have density rho=1/2, and blocks of the same size, yielding a one dimensional space of
    model. We then look at the power of the two test under one mode in this space as the null, and another as the
    alternative. As the asymptotic power is 1 in both case, we instead use Pitman asymptotic efficiency (PAE).

## Using the code
First you create you graphs of interest, here all connected graphs with at most 3 edges and set the experiment:
```python
import numpy as np
from moment_cumulant_pae import experiment_build
from graph_classes import graph_from_edgelist

edge = graph_from_edgelist([(0,1)])
cherry = graph_from_edgelist([(0,1), (1,2)])
triangle = graph_from_edgelist([(0, 1), (1, 2), (2, 0)])
path = graph_from_edgelist([(0,1), (1,2), (2, 3)])
star = graph_from_edgelist([(0,1), (0,2), (0, 3)])
experiment = experiment_build([edge, cherry, triangle, path, star])

np.save('experiment_all_3.npy', experiment)
```
This can take some time, as `experiment_build` will need to construct the formulas of cumulants as linear combinations 
of moments and formulas to compute the covariance of any moment estimate.   

Now you can compute the PAE ratio matrix as follows, say for varied graph sizes in the samples:
```python
import numpy as np
from moment_cumulant_pae import experiment_run
experiment = np.load('experiment_all_3.npy', allow_pickle=True)

x10, y10, m10 = experiment_run(experiment, n=11, k=20)
x100, y100, m100 = experiment_run(experiment, n=100, k=20)
x1000, y1000, m1000 = experiment_run(experiment, n=1000, k=20)
```

Now you can plot the output as follows:
```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=80)
cs = ax.contourf(*np.meshgrid(x10, y10), m10, cmap="rainbow")
cbar = fig.colorbar(cs)
ax.set_title('Pitman asymptotic power ratio: moment/cumulants')
ax.set_xlabel('Null')
ax.set_ylabel('Alternative')
plt.show()
```

The output is the ratio of cumulant PAE/ moment PAE, so that
* values larger than 1 indicates that cumulants beat moments
* values smaller than 1 indicates that moments beat cumulants

In the above, you will find that moments beat cumulants all the time for n <= 100, but that cumulants around half of the
time for larger graphs. 