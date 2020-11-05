from typing import Tuple

import torch
import torch.nn as nn


class UKFCell(nn.Module):
    def __init__(self, batch_size: int, state_size: int, measurement_size: int):
        """
        Args:
            batch_size: number of batches
            state_size: dimension of state
            measurement_size: dimension of measurements
        """
        super(UKFCell, self).__init__()
        self.batch_size = batch_size
        self.state_size = state_size
        self.measurement_size = measurement_size

        b, n, m = batch_size, state_size, measurement_size
        self.process_noise = nn.Parameter(torch.zeros((b, ((n + 1) * n) // 2)),
                                          requires_grad=True)
        self.measurement_noise = nn.Parameter(torch.zeros((b, ((m + 1) * m) // 2)),
                                              requires_grad=True)

    def motion_model(self, state: torch.Tensor) -> torch.Tensor:
        """
        Applies motion model to batches of sigma points

        Applies motion model to b batches of n sigma points, where each sigma point (state)
        has dimensionality m (typically: n = 2 * m + 1).

        Args:
            state: sigma points as (b, m, n) tensors

        Returns:
            Advanced states
        """
        return state

    def measurement_model(self, state: torch.Tensor) -> torch.Tensor:
        """
        Applies measurement model to batches of sigma points

        Predict measurements from b batches of n sigma points, where each sigma point (state)
        has dimensionality m (typically: n = 2 * m + 1).

        Args:
            state: sigma points as (b, m, n) tensors

        Returns:
            Predicted measurements
        """
        return state

    @staticmethod
    def tril_square(x: torch.Tensor, n: int) -> torch.Tensor:
        """
        Batched squares of triangular matrices

        The square of the (triangular) matrix L is defined as L @ L.T. The ordering of the elements
        in x has to obey those of `torch.tril_indices`.

        Args:
            x: batched n * (n + 1) / 2 elements of the batched (n, n) triangular matrices L
            n: n

        Returns:
            Batched product L @ L.T
        """
        b, _ = x.shape
        idx = torch.tril_indices(n, n)
        y = torch.zeros((b, n, n))
        y[:, idx[0], idx[1]] = x
        return torch.matmul(y, y.transpose(1, 2))

    def get_sigma_points(self,
                         mu: torch.Tensor,
                         cov: torch.Tensor,
                         *,
                         kappa: float) -> torch.Tensor:
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

        cols = torch.cholesky(cov)
        w = torch.sqrt(torch.tensor(n + kappa))

        dx = torch.zeros(b, n, 2 * n + 1)
        dx[:, :, 1:n + 1] = cols * w
        dx[:, :, n + 1:] = -cols * w

        return mu.unsqueeze(2) + dx

    def get_weights(self, *, n: int, kappa: float) -> torch.Tensor:
        """
        Returns 2n+1 weights

        If used in unscented transforms the value of kappa should be the same as used for estimating
        the sigma points.
        Weights are estimated according to:
         - w[0] = kappa / (n + kappa)
         - w[i] = .5 / (n + kappa), i = 1,...,2n+1

        Args:
            n: n as used above
            kappa: kappa as used above

        Returns:
            2n+1 weights

        """
        w = torch.ones(2 * n + 1) / 2. / (n + kappa)
        w[0] *= 2. * kappa
        return w

    def forward(self,
                measurement: torch.Tensor,
                state: torch.Tensor,
                state_cov: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        UKF step

        State and state covariance are propagated according to a motion and measurement model, new
        measurement values are predicted, compared with actual measurements, and eventually used to
        update the state and the state covariance.

        Args:
            measurement: batch of new measurements as (b, m) tensor
            state: batch of current state as (b, n) tensor
            state_cov: batch of current state covariances as (b, n, n) tensor

        Returns:
            Batched predicted measurements, updated states, and updated state covariances

        Notes:
            Below listed are the shapes of the used tensors where (b, n, m) refer to the batch size,
            the state dimension, and the dimensionality of the measured features, respectively:
             -  measurement: (b, m)
             -        state: (b, n)
             -    state_cov: (b, n, n)
             -            w: (1, 1, 2 * n + 1)
             - sigma_points: (b, n, 2 * n + 1)
             -           xs: (b, n, 2 * n + 1)
             -            x: (b, n)
             -        x_res: (b, n, 2 * n + 1)
             -        x_cov: (b, n, n)
             -           ys: (b, m, 2 * n + 1)
             -            y: (b, m)
             -        y_res: (b, m, 2 * n + 1)
             -        y_cov: (b, m, m)
             -        s_res: (b, n, 2 * n + 1)
             -       cov_sy: (b, n, m)
             -         gain: (b, n, m)
        """
        process_noise_cov = self.tril_square(self.process_noise, self.state_size)
        measurement_noise_cov = self.tril_square(self.measurement_noise, self.measurement_size)

        kappa = 3. - self.state_size
        w = self.get_weights(n=self.state_size, kappa=kappa).unsqueeze(0).unsqueeze(1)

        # compute sigma points
        sigma_points = self.get_sigma_points(state, state_cov, kappa=kappa)

        # propagate sigma points
        xs = self.motion_model(sigma_points)

        # compute predicted mean and covariance
        x = torch.sum(w * xs, dim=2)
        x_res = xs - x.unsqueeze(2)
        x_cov = torch.matmul(w * x_res, x_res.transpose(1, 2)) + process_noise_cov

        # update sigma points
        sigma_points = self.get_sigma_points(x, x_cov, kappa=kappa)

        # predict measurements
        ys = self.measurement_model(sigma_points)

        # estimate mean and covariance of predicted measurements
        y = torch.sum(w * ys, dim=2)
        y_res = ys - y.unsqueeze(2)
        y_cov = torch.matmul(w * y_res, y_res.transpose(1, 2)) + measurement_noise_cov

        # compute cross-covariance and Kalman gain
        s_res = sigma_points - x.unsqueeze(2)
        cov_sy = torch.matmul(w * s_res, y_res.transpose(1, 2))
        gain = torch.matmul(cov_sy, y_cov.inverse())

        # correct state and state covariance
        new_state = x + torch.matmul(gain, (measurement - y).unsqueeze(2)).squeeze(2)
        new_state_cov = x_cov - torch.matmul(gain, torch.matmul(y_cov, gain.transpose(1, 2)))

        return y, new_state, new_state_cov
