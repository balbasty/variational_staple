import numpy as np
from scipy.special import logsumexp, softmax, digamma, gammaln


def staple(obs, prior=None, max_iter=1000, tol=1e-5):
    """Classic STAPLE

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

    References
    ----------
    "Simultaneous Truth and Performance Level Estimation (STAPLE):
        An Algorithm for the Validation of Image Segmentation"
    Warfield, Zou & Wells
    IEEE TMI (2004)

    """
    # init obs: convert to one-hot
    if np.issubdtype(obs.dtype, np.integer):
        nb_obs, nb_raters = obs.shape
        nb_classes = obs.max() + 1
        onehot = np.zeros([nb_obs, nb_raters, nb_classes])
        np.put_along_axis(onehot, obs[:, :, None], 1, axis=-1)
        obs = onehot
    nb_obs, nb_raters, nb_classes = obs.shape

    # init log-prior: uniform
    learn_prior = False
    if prior is None:
        learn_prior = True
        prior = np.ones(nb_classes)
    prior = np.broadcast_to(np.asarray(prior), [nb_classes])
    prior = prior / nb_classes
    prior = np.log(prior)

    # init perf: softmax(-5, 5)
    perf = np.eye(nb_classes)
    perf = (perf * 2 - 1) * 5
    perf = np.broadcast_to(perf, [nb_raters, nb_classes, nb_classes])
    perf = softmax(perf, axis=1)

    loss = float('inf')
    for n_iter in range(max_iter):
        loss_prev = loss

        # E step
        posterior = prior + np.einsum('nrk,rkl->nl', obs, np.log(perf))
        loss = -np.mean(logsumexp(posterior, axis=1))
        posterior = softmax(posterior, axis=1)

        # M step: performance matrix
        perf = np.einsum('nrk,nl->rkl', obs, posterior)
        perf = softmax(np.log(perf + 1e-5), axis=1)

        # M step: class frequency
        if learn_prior and (n_iter + 1) % 10 == 0:
            prior = np.log(np.mean(posterior, axis=0))

        if loss_prev - loss < tol:
            break

    prior = softmax(prior + 1e-5)

    return perf, posterior, prior


def vstaple(obs, dirichlet=(0.8, 10), prior=None, max_iter=1000, tol=1e-5):
    """Variational STAPLE (with performance prior)

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

    References
    ----------
    "Incorporating Priors on Expert Performance Parameters for
        Segmentation Validation and Label Fusion:
        A Maximum a Posteriori STAPLE"
    Commowick & Warfield
    MICCAI (2010)

    """
    # init obs: convert to one-hot
    if np.issubdtype(obs.dtype, np.integer):
        nb_obs, nb_raters = obs.shape
        nb_classes = obs.max() + 1
        onehot = np.zeros([nb_obs, nb_raters, nb_classes])
        np.put_along_axis(onehot, obs[:, :, None], 1, axis=-1)
        obs = onehot
    nb_obs, nb_raters, nb_classes = obs.shape

    # init dirichlet: reshape to diagonal [R, K, K]
    dirichlet, df = dirichlet
    dirichlet = np.broadcast_to(np.asarray(dirichlet), [nb_raters])[:, None, None]
    df = np.broadcast_to(np.asarray(df), [nb_raters])[:, None, None]
    I = np.eye(nb_classes)
    dirichlet = I * dirichlet + (1 - I) * (1 - dirichlet) / (nb_classes - 1)
    dirichlet *= df
    sumdirichlet = np.sum(dirichlet, axis=1, keepdims=True)

    # init log-prior: uniform
    learn_prior = False
    if prior is None:
        learn_prior = True
        prior = np.ones(nb_classes)
    prior = np.broadcast_to(np.asarray(prior), [nb_classes])
    prior = prior / nb_classes
    prior = np.log(prior)

    # init perf: form prior
    logperf = digamma(dirichlet) - digamma(sumdirichlet)
    perf = dirichlet / sumdirichlet

    loss = float('inf')
    for n_iter in range(max_iter):
        loss_prev = loss

        # variational E step: true classification
        posterior = prior + np.einsum('nrk,rkl->nl', obs, logperf)
        loss = -np.sum(logsumexp(posterior, axis=1))
        posterior = softmax(posterior, axis=1)

        # variational E step: performance matrix
        perf = dirichlet + np.einsum('nrk,nl->rkl', obs, posterior)
        sumperf = np.sum(perf, axis=1, keepdims=True)
        logperf = digamma(perf) - digamma(sumperf)

        # KL between Dirichlet distributions
        loss += np.sum(gammaln(sumperf)) - np.sum(gammaln(sumdirichlet))
        loss += np.sum(gammaln(dirichlet)) - np.sum(gammaln(perf))
        loss += np.sum((perf - dirichlet) * logperf)

        # M step: class frequency
        if learn_prior and (n_iter + 1) % 10 == 0:
            prior = np.log(np.mean(posterior, axis=0))

        if loss_prev - loss < tol * len(obs):
            break

    perf /= sumperf
    prior = softmax(prior + 1e-5)

    return perf, posterior, prior

