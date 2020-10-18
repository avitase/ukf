import numpy as np
import torch


def sigma_points(mu, cov, *, kappa):
    """
    Calculates sigma points

    Sigma points are estimated according to:
     - x[:, 0] = mu
     - x[:, i] = mu + sqrt(n + kappa) * col_i(L), i = 1,...,n
     - x[:, i+n] = mu - sqrt(n + kappa) * col_i(L), i = 1,...,n
    with cov = L @ L^T

    Args:
        mu: batched mean values
        cov: batched covariance matrices
        kappa: kappa as used above

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


def unscented_transform(func, *, mu, cov, kappa=None):
    """
    Unscented transform of mean and covariance

    Unscented transform of mean and covariance is estimated according to:
     - y[:, i] = func(sigma_point[i]), i=0,...,2n
     - mu = sum_i(alpha[i] * y[:, i])
     - cov = sum_i(alpha[i] * (y[:, i] - mu) @ (y[:, i] - mu)^T)
    with weights:
     - alpha[0] = kappa / (n + k)
     - alpha[i] = .5 / (n + k), i=1,...,2n

    Args:
        mu: batched mean values
        cov: batched covariance matrices
        func: transform function
        kappa: kappa as used above (default value: 3 - n)

    Returns:
        Batched unscented transform of means and covariances
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
