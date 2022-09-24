"""
Extension of STAPLE that models correlations between raters
"""
import numpy as np
from scipy.special import logsumexp, softmax, digamma, gammaln


def vcstaple(obs, dirichlet=(5, 10), prior=None, max_iter=1000, tol=1e-5):
    """Variational C-STAPLE (raters covariance + performance prior)

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
    # init obs: convert to one-hot
    if np.issubdtype(obs.dtype, np.integer):
        nb_obs, nb_raters = obs.shape
        nb_classes = obs.max() + 1
        onehot = np.zeros([nb_obs, nb_raters, nb_classes])
        np.put_along_axis(onehot, obs[:, :, None], 1, axis=-1)
        obs = onehot
    nb_obs, nb_raters, nb_classes = obs.shape
    nb_perf = (nb_raters * (nb_raters + 1)) // 2

    # init dirichlet: reshape to diagonal [R, K, R, K, K]
    logperf00, df0 = dirichlet
    logperf00 = np.asarray(logperf00)
    df0 = np.asarray(df0)
    logperf0 = np.zeros([nb_raters, nb_classes, nb_raters, nb_classes,
                         nb_classes])
    for k in range(nb_classes):
        logperf0[range(nb_raters), k, range(nb_raters), k, k] = logperf00

    logz0 = lognorm_pairwise(logperf0)
    logperf0 -= logz0 / nb_perf
    perf0 = np.exp(logperf0)
    perf0 *= df0

    # init log-prior: uniform
    learn_prior = False
    if prior is None:
        learn_prior = True
        prior = np.ones(nb_classes)
    prior = np.broadcast_to(np.asarray(prior), [nb_classes])
    prior = prior / nb_classes
    prior = np.log(prior)

    # init perf: from prior
    df = df0 + nb_obs
    perf = perf0 * ((df0 + nb_obs) / df)
    logperf = digamma(perf) - digamma(perf.reshape([-1, perf.shape[-1]]).sum(0))

    loss = float('inf')
    for n_iter in range(max_iter):
        loss_prev = loss

        # variational E step: true classification
        posterior = prior + np.einsum('nri,risjk,nsj->nk', obs, logperf, obs)
        loss = -np.sum(logsumexp(posterior, axis=1))
        posterior = softmax(posterior, axis=1)

        # variational E step: performance matrix
        perf = perf0 + np.einsum('nri,nk,nsj->risjk', obs, posterior, obs)
        df = perf.reshape([-1, perf.shape[-1]]).sum(0)
        logperf = digamma(perf) - digamma(df)

        # KL between Dirichlet distributions
        loss += (perf - perf0) * (digamma(perf) - digamma(df))
        loss += gammaln(perf0) - gammaln(perf)
        loss = np.sum(loss) + np.sum(digamma(df) - digamma(df0))

        # M step: class frequency
        if learn_prior and (n_iter + 1) % 10 == 0:
            prior = np.log(np.mean(posterior, axis=0))

        if loss_prev - loss < tol * len(obs):
            break

    perf = np.log(perf)
    logz = lognorm_pairwise(perf)
    perf -= logz / nb_perf
    perf = np.exp(perf)
    prior = softmax(prior + 1e-5)

    return perf, posterior, prior


def lognorm_pairwise(logp, k=tuple(), sumlogp=0):
    """Compute log normalization term in the pairwise multinomial case

    Parameters
    ----------
    logp : (R, K, R, K, ...) array
        Array of unnormalized symmetric joint probabilities.
        (symmetric means p[r1, k, r2, l] == p[r2, l, r1, k])

    Returns
    -------
    logz : (...) array
        Normalization term

    """
    nr, nk, nr, nk, *batch = logp.shape
    r = len(k)
    if r == nr:
        return [sumlogp]

    z = []
    sumlogp0 = sumlogp
    for k1 in range(nk):
        sumlogp = sumlogp0 + logp[r, k1, r, k1]       # diagonal term
        for r0, k0 in enumerate(k):
            sumlogp = sumlogp + logp[r, k1, r0, k0]   # lower pairwise terms
        z += lognorm_pairwise(logp, (*k, k1), sumlogp)
    if r == 0:
        z = logsumexp(z, axis=0)
    return z


def norm_pairwise(prob, k=tuple(), prodp=1, func=None):
    """Compute normalization term in the pairwise multinomial case

    Parameters
    ----------
    prob : (R, K, R, K, ...) array
        Array of unnormalized symmetric joint probabilities.
        (symmetric means p[r1, k, r2, l] == p[r2, l, r1, k])

    Returns
    -------
    z : (...) array
        Normalization term

    """
    nr, nk, nr, nk, *batch = prob.shape
    r = len(k)
    if r == nr:
        return [prodp]

    z = []
    prodp0 = prodp
    for k1 in range(nk):
        prodp = prodp0 * prob[r, k1, r, k1]                  # diagonal term
        for r0, k0 in enumerate(k):
            prodp = prodp * prob[r, k1, r0, k0]             # lower pairwise terms
        z += norm_pairwise(prob, (*k, k1), prodp)
    if r == 0:
        if func:
            z = map(func, z)
        z = np.sum(z, axis=0)
    return z


def marginal_pairwise(prob, indices=None, k=tuple(), prod=1, z=None):
    """Compute marginal probability in the pairwise multinomial case

    Parameters
    ----------
    prob : (R, K, R, K, ...) array
        Array of normalized symmetric joint probabilities.
        (symmetric means p[r1, k, r2, l] == p[r2, l, r1, k])
    indices : int or list[int], optional
        Indices whose joint marginal to compute
        If not provided, compute marginal of each rater independently

    Returns
    -------
    z : (K**len(indices), ...) array
        Marginal probability
        If indices is None, return a (K, ...) array containing all
        marginal probabilities

    """
    nr, nk, nr, nk, *batch = prob.shape

    if indices is None:
        return np.stack([marginal_pairwise(prob, i) for i in range(nr)])

    r = len(k)
    if r == nr:
        return [prod]

    if not isinstance(indices, (list, tuple, range)):
        indices = [indices]
    indices = list(indices)

    # reorder such that marginalized indices are last
    if r == 0:
        marginalized_indices = [i for i in range(nr) if i not in indices]
        all_indices = indices + marginalized_indices
        prob = prob[all_indices][:, :, all_indices]

    if r < len(indices):
        if z is None:
            z = np.zeros([nk] * len(indices) + list(batch))
    else:
       z = []

    prod0 = prod
    for k1 in range(nk):
        prod = prod0 * prob[r, k1, r, k1]       # diagonal term
        for r0, k0 in enumerate(k):
            prod = prod * prob[r, k1, r0, k0]   # lower pairwise terms
        if r < len(indices) - 1:
            marginal_pairwise(prob, indices, (*k, k1), prod, z[k1])
        elif r == len(indices) - 1:
            z[k1] = np.sum(marginal_pairwise(prob, indices, (*k, k1), prod), axis=0)
        else:
            z += marginal_pairwise(prob, indices, (*k, k1), prod)
    return z


def covariance_pairwise(prob, i=None, j=None):
    """Compute raters covariance in the pairwise multinomial case

    Parameters
    ----------
    prob : (R, K, R, K, ...) array
        Array of normalized symmetric joint probabilities.
        (symmetric means p[r1, k, r2, l] == p[r2, l, r1, k])
    i, j=i : int
        Indices of the raters whose covariance to compute

    Returns
    -------
    cov : (K, K, ...) array
        Covariance
        If i, j are None, return all possible covariances in a
        (R, R, K, K, ...) array

    """
    nr, nk, nr, nk, *batch = prob.shape

    if i is None:
        return np.stack([np.stack([covariance_pairwise(prob, i, j)
                                  for j in range(nr)]) for i in range(nr)])

    if j is None:
        j = i
    if j == i:
        # variance
        p = marginal_pairwise(prob, i)
        c = - p[:, None] * p[None, :]       # outer product
        c[range(nk), range(nk)] += p        # diagonal
        return c
    else:
        # covariance
        joint = marginal_pairwise(prob, [i, j])
        margi = marginal_pairwise(prob, i)
        margj = marginal_pairwise(prob, j)
        c = joint - margi[:, None] * margj[None, :]
        return c


def correlation_pairwise(prob):
    """Compute raters correlation in the pairwise multinomial case

    Parameters
    ----------
    prob : (R, K, R, K, ...) array
        Array of normalized symmetric joint probabilities.

    Returns
    -------
    var : (R, ...) array
        Variance
    corr : (R, R, ...) array
        Correlation

    """
    nr, nk, nr, nk, *batch = prob.shape
    cov = covariance_pairwise(prob)[:, :, range(nk), range(nk)].sum(2)
    var = cov[range(nr), range(nr)]
    corr = cov / (np.sqrt(var)[:, None] * np.sqrt(var)[None, :])
    return var, corr
