import numpy as np
import torch


def sigma_points(mu, cov, *, kappa):
    b, n = mu.shape
    L = torch.cholesky(cov)
    X = mu.unsqueeze(b).expand(b, n, 2 * n + 1)

    w = np.sqrt(n - kappa)

    dX = torch.zeros_like(X)
    dX[:, :, 1:n + 1] = L * w
    dX[:, :, n + 1:] = -L * w

    return X + dX


def unscented_transform(mu, cov, *, func, kappa=None):
    b, n, m = cov.shape
    assert m == n
    assert mu.ndim == 2
    assert mu.shape == (b, n)

    if kappa is None:
        kappa = 3 - n

    alpha = torch.ones(b, n, 2 * n + 1) / 2. / (n + kappa)
    alpha[:, :, 0] *= 2. * kappa

    X = sigma_points(mu, cov, kappa=kappa)
    Y = func(X)
    assert Y.shape == (b, n, 2 * n + 1)

    mu2 = torch.sum(alpha * Y, dim=2)
    assert mu2.ndim == 2
    assert mu2.shape == mu.shape

    R = Y - mu2.unsqueeze(2).expand(2, n, 2 * n + 1)
    cov2 = torch.matmul(alpha * R, R.transpose(1, 2))
    assert cov2.ndim == 3
    assert cov2.shape == cov.shape

    return mu2, cov2
