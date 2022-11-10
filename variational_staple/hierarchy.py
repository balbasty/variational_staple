import numpy as np
from numpy import ma
from scipy.special import logsumexp, softmax, digamma, gammaln


def hstaple(obs, parents, perf_prior=(1, 10), child_prior=(1, 10),
            parent_prior=(0, 10), max_iter=1000, tol=1e-5):
    """Hierarchical STAPLE

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
    # init obs: convert to one-hot
    if np.issubdtype(obs.dtype, np.integer) and obs.ndim < 3:
        mask = None
        if isinstance(obs, np.ma.MaskedArray):
            mask = obs.mask
            obs = obs.data
        nb_obs, nb_raters = obs.shape
        nb_classes = obs.max() + 1
        onehot = np.zeros([nb_obs, nb_raters, nb_classes])
        np.put_along_axis(onehot, obs[:, :, None], 1, axis=-1)
        obs = onehot
        if mask is not None:
            obs[mask] = float('nan')
    if np.any(~np.isfinite(obs)):
        mask = ~np.isfinite(obs)
        obs = ma.masked_array(obs, mask=mask)
    if isinstance(obs, np.ma.MaskedArray):
        mask = obs.mask
        obs = np.copy(obs.data)
        obs[mask] = 0

    if np.issubdtype(parents.dtype, np.integer) and parents.ndim < 3:
        nb_obs = len(obs)
        nb_parents = parents.max() + 1
        onehot = np.zeros([nb_obs, nb_parents])
        np.put_along_axis(onehot, parents[:, None], 1, axis=-1)
        parents = onehot

    nb_obs, nb_raters, nb_classes = obs.shape
    _, nb_parents = parents.shape

    # init dirichlet: reshape to diagonal [R, K, K]
    alpha0, df_alpha0 = init_dirichlet_rater(*perf_prior, nb_classes, nb_raters)
    beta0, df_beta0 = init_dirichlet_variability(*child_prior, nb_classes)
    gamma0, df_gamma0 = init_dirichlet_frequency(*parent_prior, nb_classes)

    log_alpha = digamma(alpha0 * df_alpha0) - digamma(df_alpha0)
    log_beta = digamma(beta0 * df_beta0) - digamma(df_beta0)
    log_gamma = digamma(gamma0 * df_gamma0) - digamma(df_gamma0)

    alpha, df_alpha = alpha0, df_alpha0 + np.sum(np.sum(obs, 2), 0)[:, None, None]
    beta, df_beta = beta0, df_beta0 + len(obs)
    gamma, df_gamma = gamma0, df_gamma0 + nb_parents

    parent_posterior = np.broadcast_to(gamma0, [nb_parents, nb_classes])

    # init loss
    loss = 0
    loss += np.sum(gammaln(df_alpha)) - np.sum(gammaln(df_alpha0))
    loss -= np.sum(gammaln(alpha)) - np.sum(gammaln(alpha0))
    loss += np.sum((alpha - alpha0) * log_alpha)
    loss += np.sum(gammaln(df_beta)) - np.sum(gammaln(df_beta0))
    loss -= np.sum(gammaln(beta)) - np.sum(gammaln(beta0))
    loss += np.sum((beta - beta0) * log_beta)
    loss += np.sum(gammaln(df_gamma)) - np.sum(gammaln(df_gamma0))
    loss -= np.sum(gammaln(gamma)) - np.sum(gammaln(gamma0))
    loss += np.sum((gamma - gamma0) * log_gamma)

    loss_prev = float('inf')
    for n_iter in range(max_iter):

        # variational E step: true classification (child level)
        child_posterior = np.einsum('nrk,rkl->nl', obs, log_alpha)
        child_posterior += np.einsum('np,pk,lk->nl', parents, parent_posterior, log_beta)
        loss -= np.sum(logsumexp(child_posterior, axis=1))
        child_posterior = softmax(child_posterior, axis=1)

        # variational E step: true classification (parent level)
        parent_posterior = np.einsum('nk,np,kl->pl', child_posterior, parents, log_beta)
        parent_posterior += log_gamma
        loss -= np.sum(logsumexp(parent_posterior, axis=1))
        parent_posterior = softmax(parent_posterior, axis=1)

        # term counted twice
        loss += np.einsum('nk,np,pl,kl->', child_posterior, parents, parent_posterior, log_beta)

        # print(n_iter, loss, (loss_prev - loss) / len(obs))
        if abs(loss_prev - loss) < tol * len(obs):
            break
        loss_prev, loss = loss, 0

        # variational E step: performance matrix
        alpha = np.einsum('nrk,nl->rkl', obs, child_posterior)
        alpha += alpha0 * df_alpha0
        df_alpha = np.sum(alpha, axis=1, keepdims=True)
        log_alpha = digamma(alpha) - digamma(df_alpha)

        # KL between Dirichlet distributions
        loss += np.sum(gammaln(df_alpha)) - np.sum(gammaln(df_alpha0))
        loss -= np.sum(gammaln(alpha)) - np.sum(gammaln(alpha0))
        loss += np.sum((alpha - alpha0) * log_alpha)

        # variational E step: variability matrix
        beta = np.einsum('nk,np,pl->kl', child_posterior, parents, parent_posterior)
        beta += beta0 * df_beta0
        df_beta = np.sum(beta, axis=0, keepdims=True)
        log_beta = digamma(beta) - digamma(df_beta)

        # KL between Dirichlet distributions
        loss += np.sum(gammaln(df_beta)) - np.sum(gammaln(df_beta0))
        loss -= np.sum(gammaln(beta)) - np.sum(gammaln(beta0))
        loss += np.sum((beta - beta0) * log_beta)

        # variational E step: class frequency
        gamma = np.sum(parent_posterior, 0)
        gamma += gamma0 * df_gamma0
        df_gamma = np.sum(gamma)
        log_gamma = digamma(gamma) - digamma(df_gamma)

        # KL between Dirichlet distributions
        loss += np.sum(gammaln(df_gamma)) - np.sum(gammaln(df_gamma0))
        loss -= np.sum(gammaln(gamma)) - np.sum(gammaln(gamma0))
        loss += np.sum((gamma - gamma0) * log_gamma)

    alpha /= df_alpha
    beta /= df_beta
    gamma /= df_gamma

    return alpha, child_posterior, parent_posterior, beta, gamma


def init_dirichlet_rater(alpha, df, nb_classes, nb_raters):
    """Initialize Dirichlet prior for the performance matrix

    Parameters
    ----------
    alpha : float or (nb_raters,) array
    df : float or (nb_raters,) array
    nb_classes : int
    nb_raters : int

    Returns
    -------
    alpha : (nb_raters, nb_classes, nb_classes) array
    df : (nb_raters, 1, nb_classes) array

    """
    alpha, df = np.asarray(alpha), np.asarray(df)
    if alpha.ndim <= 1:
        alpha = np.broadcast_to(alpha, [nb_raters])[:, None, None]
        alpha = alpha * np.eye(nb_classes)
    else:
        alpha = np.broadcast_to(alpha, [nb_raters, nb_classes, nb_classes])
    alpha = np.exp(alpha)
    alpha /= np.sum(alpha, axis=-2, keepdims=True)
    df = np.broadcast_to(df, [nb_raters])[:, None, None]
    df = np.broadcast_to(df, [nb_raters, 1, nb_classes])
    return alpha, df


def init_dirichlet_variability(alpha, df, nb_classes, nb_raters=0):
    """Initialize Dirichlet prior for the variability matrix

    Parameters
    ----------
    alpha : float or (nb_raters,) array
    df : float or (nb_raters,) array
    nb_classes : int
    nb_raters : int

    Returns
    -------
    alpha : ([nb_raters], nb_classes, nb_classes) array
    df : ([nb_raters], 1, nb_classes) array

    """
    joint = nb_raters == 0
    nb_raters = nb_raters or 1
    alpha, df = np.asarray(alpha), np.asarray(df)
    if alpha.ndim <= 1:
        alpha = np.broadcast_to(alpha, [nb_raters])[:, None, None]
        alpha = alpha * np.eye(nb_classes)
    else:
        alpha = np.broadcast_to(alpha, [nb_raters, nb_classes, nb_classes])
    alpha = np.exp(alpha)
    alpha /= np.sum(alpha, axis=-2, keepdims=True)
    df = np.broadcast_to(df, [nb_raters])[:, None, None]
    df = np.broadcast_to(df, [nb_raters, 1, nb_classes])
    if joint:
        alpha, df = alpha[0], df[0]
    return alpha, df


def init_dirichlet_frequency(alpha, df, nb_classes):
    """Initialize Dirichlet prior for class frequencies

    Parameters
    ----------
    alpha : float or (nb_classes,) array
    df : float
    nb_classes : int

    Returns
    -------
    alpha : (nb_classes,) array
    df : () array

    """
    alpha, df = np.asarray(alpha), np.asarray(df)
    alpha = np.broadcast_to(alpha, [nb_classes])
    alpha = np.exp(alpha)
    alpha /= np.sum(alpha)
    return alpha, df
