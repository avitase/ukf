import torch
import numpy as np


def get_sigma_points(mu, cov, *, kappa=None):
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
        kappa: kappa as used above (default value: 3 - n)

    Returns:
        Sigma points as batched (n, 2 * n + 1) matrix
    """

    b, n = mu.shape

    if kappa is None:
        kappa = 3. - n

    L = torch.cholesky(cov)
    X = mu.unsqueeze(2).expand(b, n, 2 * n + 1)

    w = np.sqrt(n + kappa)

    dX = torch.zeros_like(X)
    dX[:, :, 1:n + 1] = L * w
    dX[:, :, n + 1:] = -L * w

    return X + dX


def get_weights(*, b, n, kappa):
    w = torch.ones(b, 1, 2 * n + 1) / 2. / (n + kappa)
    w[:, :, 0] *= 2. * kappa
    return w


def ukf_step(*, motion_model, measurement_model, state, state_cov, process_noise, measurement_noise,
             kappa=None):
    b, n = state.shape
    if kappa is None:
        kappa = 3. - n

    weights = get_weights(b=b, n=n, kappa=kappa)

    # compute sigma points
    sigma_points = get_sigma_points(state, state_cov, kappa=kappa)

    # propagate sigma points
    xs = motion_model(sigma_points)

    # compute predicted mean and covariance
    wx = weights.expand(xs.shape)
    x = torch.sum(wx * xs, dim=2)
    res_x = xs - x.unsqueeze(2).expand(*xs.shape)
    cov_x = torch.matmul(wx * res_x, res_x.transpose(1, 2)) + process_noise

    # update sigma points
    sigma_points = get_sigma_points(x, cov_x, kappa=kappa)

    # predict measurements
    ys = measurement_model(sigma_points)

    # estimate mean and covariance of predicted measurements
    wy = weights.expand(ys.shape)
    y = torch.sum(wy * ys, dim=2)
    res_y = ys - y.unsqueeze(2).expand(*ys.shape)
    cov_y = torch.matmul(wy * res_y, res_y.transpose(1, 2)) + measurement_noise

    # compute cross-covariance and Kalman gain
    res_s = sigma_points - x.unsqueeze(2).expand(*sigma_points.shape)
    cov_sy = torch.matmul(wx * res_s, res_y.transpose(1, 2))
    gain = torch.matmul(cov_sy, cov_y.inverse())

    return x, y, cov_x, cov_y, gain
