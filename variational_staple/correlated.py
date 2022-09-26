"""
Extension of STAPLE that models correlations between raters
"""
import numpy as np
from scipy.special import logsumexp, softmax


def cstaple(obs, dirichlet=(5, 10), prior=None, max_iter=1000, tol=1e-5):
    """C-STAPLE (raters co-occurrence + performance prior)

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
        df, nr = obs.shape
        nk = obs.max() + 1
        onehot = np.zeros([df, nr, nk])
        np.put_along_axis(onehot, obs[:, :, None], 1, axis=-1)
        obs = onehot
    nobs, nr, nk = obs.shape
    nperf = (nr * (nr + 1)) // 2

    # init dirichlet: reshape to diagonal [R, K, R, K, K]
    alpha, df0 = dirichlet
    alpha, df0 = np.asarray(alpha), np.asarray(df0)
    theta = np.zeros([nr, nk, nr, nk, nk])
    for k in range(nk):
        theta[range(nr), k, range(nr), k, k] = alpha

    logz = lognorm_pairwise(theta)
    logperf = theta - logz / nperf
    perf = ensure_zeros(np.exp(logperf))    # initial confusion matrix
    alpha = marginal2_pairwise(perf)        # fake observations used as prior

    # init log-prior: uniform
    learn_prior = False
    if prior is None:
        learn_prior = True
        prior = np.ones(nk)
    prior = np.broadcast_to(np.asarray(prior), [nk])
    prior = prior / nk

    loss = float('inf')
    for n_iter in range(max_iter):
        loss_prev = loss

        # E step: true classification
        logposterior = np.log(prior) - logz \
                       + np.einsum('nri,risjk,nsj->nk', obs, theta, obs)
        loss = -np.sum(logsumexp(logposterior, axis=1))
        posterior = softmax(logposterior, axis=1)
        df = posterior.sum(axis=0)

        # M step: performance matrix
        g0 = df0 * alpha + np.einsum('nri,nk,nsj->risjk', obs, posterior, obs)
        g0 = g0 / (df0 + df)
        subloss = np.sum((df + df0) * logz) / np.sum(df + df0)
        subloss -= np.dot(g0.flatten(), theta.flatten())
        for _ in range(100):
            thetaprev, logzprev, sublossprev = theta, logz, subloss
            # gradient == E[xi xj] - mean(obs[xni xnj])
            delta = marginal2_pairwise(perf) - g0
            # hessian is majorized by identity
            theta -= 0.1 * delta
            # update normalization term
            logz = lognorm_pairwise(theta)
            logperf = theta - logz / nperf
            perf = ensure_zeros(np.exp(logperf.clip(None, 512)))
            # check subloss
            subloss = np.sum((df + df0) * logz) / np.sum(df + df0)
            subloss -= np.dot(g0.flatten(), theta.flatten())
            if sublossprev - subloss < 2 * tol:
                if sublossprev < subloss:
                    theta, logz = thetaprev, logzprev
                logperf = theta - logz / nperf
                perf = ensure_zeros(np.exp(logperf.clip(None, 512)))
                break

        # loss: dirichlet prior
        loss += df0 * (np.sum(logz) - np.dot(alpha.flatten(), logperf.flatten()))

        # M step: class frequency
        if learn_prior and (n_iter + 1) % 10 == 0:
            prior = df / nobs

        # print(n_iter, loss, (loss_prev - loss) / len(obs))
        if loss_prev - loss < tol * len(obs):
            break

    return perf, posterior, prior


def ensure_zeros(prob):
    """Ensure that co-occurrence of different classes in the same rater is zero"""
    nr, nk, nr, nk, *batch = prob.shape
    for r in range(nr):
        diag = prob[r, range(nk), r, range(nk)]
        prob[r, :, r, :] = 0
        prob[r, range(nk), r, range(nk)] = diag
    return prob


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


def marginal2_pairwise(prob):
    nr, nk, nr, nk, *batch = prob.shape
    out = np.zeros_like(prob)
    for i in range(nr):
        out[i, range(nk), i, range(nk)] = marginal_pairwise(prob, i)
        for j in range(i):
            out[i, :, j, :] = marginal_pairwise(prob, [i, j])
            out[j, :, i, :] = np.swapaxes(out[i, :, j, :], 0, 1)
    return out


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

    # reorder such that marginalized indices are last
    if r == 0:
        if not isinstance(indices, (list, tuple, range)):
            indices = [indices]
        indices = list(indices)
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
        # rater variance (== within rater covariance)
        p = marginal_pairwise(prob, i)
        c = - p[:, None] * p[None, :]       # outer product
        c[range(nk), range(nk)] += p        # diagonal
        return c
    else:
        # rater covariance
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
