"""
Hierarchical extension of STAPLE using a mixture of confusion matrices
"""
import numpy as np
from scipy.special import logsumexp, softmax, digamma, gammaln


def mstaple(obs, nb_clusters=2, dirichlet=(0.8, 10), prior=None, max_iter=1000, tol=1e-5):
    """M-STAPLE (mixture model + performance prior)

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
    # init obs: convert to one-hot
    if np.issubdtype(obs.dtype, np.integer):
        df, nr = obs.shape
        nk = obs.max() + 1
        onehot = np.zeros([df, nr, nk])
        np.put_along_axis(onehot, obs[:, :, None], 1, axis=-1)
        obs = onehot
    nobs, nr, nk = obs.shape
    nc = nb_clusters

    # prepare dirichlet prior
    alpha, df0 = dirichlet
    alpha, df0 = np.asarray(alpha), np.asarray(df0)
    dirichlet = np.zeros([nc, nr, nk, nk])
    dirichlet[...] = (1 - alpha[:, None, None]) / (nk - 1)
    dirichlet[..., range(nk), range(nk)] = alpha[:, None]
    dirichlet *= df0[..., None, None]
    df0 = np.sum(dirichlet, axis=-2, keepdims=True)

    # init log-prior: uniform
    learn_prior = False
    if prior is None:
        learn_prior = True
        prior = np.ones(nk)
    prior = np.broadcast_to(np.asarray(prior), [nk])
    prior = prior / nk

    # init clusters
    cluster_prior = np.ones(nc) / nc
    clusters = np.ones([nobs, nc]) / nc

    # init confusion matrices
    perf = np.random.gamma(dirichlet, 1)
    perf = perf / perf.sum(-2, keepdims=True)
    df = df0 + nobs / (nc * nk)
    perf *= df
    logperf = digamma(perf * df) - digamma(df)

    loss = float('inf')
    for n_iter in range(max_iter):
        loss_prev = loss

        # variational E step: true classification
        posterior = np.log(prior) + np.einsum('nrk,crkl,nc->nl', obs, logperf, clusters)
        loss = -np.sum(logsumexp(posterior, axis=1))
        posterior = softmax(posterior, axis=1)
        # M step: classification prior
        if learn_prior:
            prior = np.mean(posterior, axis=0)

        # variational E step: clusters
        clusters = np.log(cluster_prior) + np.einsum('nrk,crkl,nl->nc', obs, logperf, posterior)
        clusters = softmax(clusters, axis=1)
        # M step: cluster prior
        cluster_prior = np.mean(clusters, axis=0)

        # KL between categorical distributions
        loss += (clusters * (np.log(clusters) - np.log(cluster_prior))).sum()

        # variational E step: performance matrix
        perf = dirichlet + np.einsum('nrk,nl,nc->crkl', obs, posterior, clusters)
        df = np.sum(perf, axis=-2, keepdims=True)
        logperf = digamma(perf) - digamma(df)

        # KL between Dirichlet distributions
        loss += np.sum(gammaln(df)) - np.sum(gammaln(df0))
        loss += np.sum(gammaln(dirichlet)) - np.sum(gammaln(perf))
        loss += np.sum((perf - dirichlet) * logperf)

        print(n_iter, loss, (loss_prev - loss) / len(obs))
        if n_iter > 2 and loss_prev - loss < tol * len(obs):
            break

    perf /= df
    return perf, posterior, prior, clusters, cluster_prior


def marginals(perf, prior):
    """Compute confusion matrix after marginalizing clusters out"""
    return np.einsum('crik,c->rik', perf, prior)


def correlations(perf, prior):
    """Compute correlation matrix after marginalizing clusters out"""
    nr = perf.shape[1]
    marg2 = np.einsum('crik,clik,c->rlk', perf, perf, prior)
    marg1 = marginals(perf, prior)
    marg2[range(nr), range(nr), :] = marg1.sum(-2)
    cov = marg2 - np.einsum('rik,lik->rlk', marg1, marg1)
    sd = np.sqrt(cov[range(nr), range(nr), :])
    corr = cov / (sd[None, :] * sd[:, None])
    return corr
