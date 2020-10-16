import numpy as np
import torch


def sigma_points(mu, cov, *, kappa):
    """
    Calculates sigma points

    Args:
        mu: batched mean values
        cov: batched covariance matrices
        kappa: +/- sqrt(N + kappa) are used as weights

    Returns:
        Sigma points as batched (n, 2 * n + 1) matrix
    """

    b, n = mu.shape
    L = torch.cholesky(cov)
    X = mu.unsqueeze(2).expand(b, n, 2 * n + 1)

    w = np.sqrt(n + kappa)

    dX = torch.zeros_like(X)
    dX[:, :, 1:n + 1] = L * w
    dX[:, :, n + 1:] = -L * w

    return X + dX


def unscented_transform(mu, cov, *, func, kappa=None):
    """
    Unscented transform of mean and covariance

    Args:
        mu: batched mean values
        cov: batched covariance matrices
        func: transform function
        kappa: kappa / (N + kappa) and .5 / (N + kappa) are used as weights

    Returns:
        Unscented transform of mean and covariance
    """

    b, n, _ = cov.shape

    if kappa is None:
        kappa = 3. - n

    alpha = torch.ones(b, n, 2 * n + 1) / 2. / (n + kappa)
    alpha[:, :, 0] *= 2. * kappa

    X = sigma_points(mu, cov, kappa=kappa)
    Y = func(X)

    mu2 = torch.sum(alpha * Y, dim=2)

    R = Y - mu2.unsqueeze(2).expand(b, n, 2 * n + 1)
    cov2 = torch.matmul(alpha * R, R.transpose(1, 2))

    return mu2, cov2
