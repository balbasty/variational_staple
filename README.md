# Variational STAPLE

This package implements label fusion algorithms based on
STAPLE (Simultaneous Truth And Performance-Level Evaluation) and its 
MAP (Maximum A Posteriori) variant.

Note that while the original STAPLE targets image segmentation, our
implementation focuses on iid samples, and therefore does not include
MRF spatial regularization or spatially local variants.

However, we implemented three novel variants:
- HSTAPLE: variant that adds hierarchical levels to model 
  correlations between subsets of observations.
- MSTAPLE: variant where each rater's confusion matrix is 
  sampled from a mixture of matrices.
- CSTAPLE: variant where raters can be correlated under a second-order 
  click model.

Note that the API is unstable and may change in the near future.

## Quick documentation

```python
staple(obs, prior=None, max_iter=1000, tol=1e-5)
"""
Classic STAPLE

Parameters
----------
obs : (N, R, K) array[float] or (N, R) array[int]
    Classification of N data points by R raters into K classes
prior : (K,) array, optional
    Prior probability of observing each class.
    By default, it is ML estimated

Returns
-------
perf : (R, K, K) array
    Performance matrix = conditional probability of a rater
    saying "red" when the truth is "green"
posterior : (N, K) array
    Posterior probability of the true classification
prior : (K,) array
    Prior probability of observing each class
"""
```

```python
vstaple(obs, dirichlet=(0.8, 10), prior=None, max_iter=1000, tol=1e-5)
"""
Variational STAPLE (with performance prior)

Notes
-----
Our implementation differs from that of Commowick and Warfield
(MAP-STAPLE) in that we use variational inference rather than a
maximum a posteriori approach.

Parameters
----------
obs : (N, R, K) array[float] or (N, R) array[int]
    Classification of N data points by R raters into K classes
dirichlet : tuple[float or (R,) array]
    Parameter of the Dirichlet prior:
        [0] probability of being correct, in 0..1
        [1] degrees of freedom, > 0 (higher = stronger prior)
prior : (K,) array, optional
    Prior probability of observing each class.
    By default, it is ML estimated

Returns
-------
perf : (R, K, K) array
    Expected performance matrix = conditional probability of a rater
    saying "red" when the truth is "green"
posterior : (N, K) array
    Posterior probability of the true classification
prior : (K,) array
    Prior probability of observing each class
"""
```

```python
mstaple(obs, nb_clusters=2, dirichlet=(0.8, 10), prior=None, max_iter=1000, tol=1e-5)
"""
M-STAPLE (mixture model + performance prior)

Parameters
----------
obs : (N, R, K) array[float] or (N, R) array[int]
    Classification of N data points by R raters into K classes
nb_clusters : int
    Number of clusters in the mixture
dirichlet : tuple[float or (R,) array, float]
    Parameter of the diagonal Dirichlet prior:
        [0] Probability of being correct, in (0, 1)
        [1] degrees of freedom > 0 (higher = stronger prior)
prior : (K,) array, optional
    Prior probability of observing each class.
    By default, it is ML estimated

Returns
-------
perf : (C, R, K, K) array
    Performance matrix = conditional probability of a rater
    saying "red" when the truth is "green"
posterior : (N, K) array
    Posterior probability of the true classification
prior : (K,) array
    Prior probability of observing each class
cluster : (N, C) array
    Cluster assignment of each data point
cluster_prior : (C,) array
    Prior probability of belonging to each cluster

"""
```

```python
hstaple(obs, parents, perf_prior=(1, 10), child_prior=(1, 10),
            parent_prior=(0, 10), max_iter=1000, tol=1e-5)
"""
Hierarchical STAPLE

This is an extension of variational STAPLE that models two levels
of "truth".

Forward model:
    - level0[n] ~ Cat(p)                        # truth at the parent/subject level
    - level1[m] | level0[n] ~ Cat(q[level0])    # truth at the child/repeat level
    - level2[r] | level1[m] ~ Cat(r[level1])    # rating
Priors:
    - p ~ Dirichlet(a)       # class frequency
    - q ~ Dirichlet(b)       # parent-to-child confusion (intra-subject variability)
    - r ~ Dirichlet(c)       # child-to-rater confusion (performance)

Parameters
----------
obs : (N, R, K) array[float] or (N, R) array[int]
    Classification of N data points by R raters into K classes.
    A masked array can be provided to encode missing values
parents : (N, P) array[float] or (N,) array[int]
    Parent of each data point in the hierarchy
perf_prior : tuple[float or (R,) array]
    Dirichlet prior over performance matrices:
        [0] log probability of being correct > 0
        [1] degrees of freedom, > 0 (higher = stronger prior, 0 = ML, inf = fixed)
child_prior : tuple[float or (P,) array], optional
    Dirichlet prior over child-to-parent confusion matrices:
        [0] log probability of being correct > 0
        [1] degrees of freedom, > 0 (higher = stronger prior, 0 = ML, inf = fixed)
parent_prior : tuple[float], optional
    Dirichlet prior over class frequencies at the parent level:
        [0] log probability of being correct > 0
        [1] degrees of freedom, > 0 (higher = stronger prior, 0 = ML, inf = fixed)

Returns
-------
perf : (R, K, K) array
    Expected performance matrix = conditional probability of a rater
    saying "red" when the truth is "green"
child_posterior : (N, K) array
    Posterior probability of the true classification at the child level
parent_posterior : (P, K) array
    Posterior probability of the true classification at the parent level
child_prior : (K, K) array
    Prior probability of observing a child score given the parent score
parent_prior : (K,) array
    Prior probability of observing each class at the parent level
"""
```

```python
cstaple(obs, dirichlet=(1, 10), prior=None, max_iter=1000, tol=1e-5)
"""
C-STAPLE (raters co-occurrence + performance prior)

Parameters
----------
obs : (N, R, K) array[float] or (N, R) array[int]
    Classification of N data points by R raters into K classes
dirichlet : tuple[float or (R,) array, float]
    Parameter of the diagonal Dirichlet prior:
        [0] Correctness score > 0
        [1] degrees of freedom > 0 (higher = stronger prior)
prior : (K,) array, optional
    Prior probability of observing each class.
    By default, it is ML estimated

Returns
-------
perf : (R, K, R, K, K) array
    Expected performance matrix = conditional joint probability
    of two raters saying ("red", "blue") when the truth is "green"
posterior : (N, K) array
    Posterior probability of the true classification
prior : (K,) array
    Prior probability of observing each class

"""
```

## References

- **"Simultaneous Truth and Performance Level Estimation (STAPLE): <br />
     An Algorithm for the Validation of Image Segmentation"** <br />
  Warfield, Zou & Wells <br />
  IEEE TMI (2004)
- **"Incorporating Priors on Expert Performance Parameters for Segmentation 
     Validation and Label Fusion: <br /> 
     A Maximum a Posteriori STAPLE"** <br />
   Commowick & Warfield <br />
   MICCAI (2010)
- **"Consolidation of Expert Ratings of Motion Artifacts using Hierarchical 
     Label Fusion"** <br />
  Balbastre & al <br />
  Unpublished (2022)
