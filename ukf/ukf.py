import numpy as np
import torch


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

    cols = torch.cholesky(cov)
    w = np.sqrt(n + kappa)

    dx = torch.zeros(b, n, 2 * n + 1)
    dx[:, :, 1:n + 1] = cols * w
    dx[:, :, n + 1:] = -cols * w

    return mu.unsqueeze(2) + dx


def get_weights(*, b, n, kappa):
    w = torch.ones(b, 1, 2 * n + 1) / 2. / (n + kappa)
    w[:, :, 0] *= 2. * kappa
    return w


def ukf_step(*, motion_model, measurement_model, state, state_cov, process_noise, measurement_noise,
             kappa=None):
    """
    UKF step

    State and state covariance are propagated according to a motion and measurement model.
    The returned values can be used together with an actual measurement to compute the corrected
    mean and covariance.

    Args:
        motion_model: function that propagates (b, n, 2 * n + 1) data according to motion model
        measurement_model: function that predicts (b, m) measurements from (b, n, 2 * n + 1) data
        state: state of type (b, n) tensor
        state_cov: state covariance of type (b, n, n) tensor
        process_noise: process noise of type (b, n, n) tensor
        measurement_noise: measurement noise of type (b, m, m) tensor
        kappa: kappa as used in get_weights (default value: 3 - n)

    Returns:
        x: predicted mean of state
        y: estimated mean of predicted measurements
        cov_x: predicted state covariance
        cov_y: estimated covariance of predicted measurements
        gain: Kalman gain

    Notes:
        Below listed are the shapes of the used tensors where (b, n, m) refer to the batch size,
        the state dimension, and the dimensionality of the measured features, respectively:
         -            w: (b, 1, 2 * n + 1)
         - sigma_points: (b, n, 2 * n + 1)
         -           xs: (b, n, 2 * n + 1)
         -            x: (b, n)
         -        res_x: (b, n, 2 * n + 1)
         -        cov_x: (b, n, n)
         -           ys: (b, m, 2 * n + 1)
         -            y: (b, m)
         -        res_y: (b, m, 2 * n + 1)
         -        cov_y: (b, m, m)
         -        res_s: (b, n, 2 * n + 1)
         -       cov_sy: (b, n, m)
         -         gain: (b, n, m)
    """
    b, n = state.shape
    if kappa is None:
        kappa = 3. - n

    w = get_weights(b=b, n=n, kappa=kappa)

    # compute sigma points
    sigma_points = get_sigma_points(state, state_cov, kappa=kappa)

    # propagate sigma points
    xs = motion_model(sigma_points)

    # compute predicted mean and covariance
    x = torch.sum(w * xs, dim=2)
    res_x = xs - x.unsqueeze(2)
    cov_x = torch.matmul(w * res_x, res_x.transpose(1, 2)) + process_noise

    # update sigma points
    sigma_points = get_sigma_points(x, cov_x, kappa=kappa)

    # predict measurements
    ys = measurement_model(sigma_points)

    # estimate mean and covariance of predicted measurements
    y = torch.sum(w * ys, dim=2)
    res_y = ys - y.unsqueeze(2)
    cov_y = torch.matmul(w * res_y, res_y.transpose(1, 2)) + measurement_noise

    # compute cross-covariance and Kalman gain
    res_s = sigma_points - x.unsqueeze(2)
    cov_sy = torch.matmul(w * res_s, res_y.transpose(1, 2))
    gain = torch.matmul(cov_sy, cov_y.inverse())

    return x, y, cov_x, cov_y, gain


def ukf_correct(y_measured, *, x, y_predicted, cov_x, cov_y, gain):
    x_new = x + torch.matmul(gain, (y_measured - y_predicted).unsqueeze(2)).squeeze(2)
    cov_x_new = cov_x - torch.matmul(gain, torch.matmul(cov_y, gain.transpose(1, 2)))
    return x_new, cov_x_new
