from typing import Tuple

import torch

import ukf


class SimpleUKFCell(ukf.UKFCell):
    def __init__(self, *, batch_size: int, log_cholesky: bool):
        super(SimpleUKFCell, self).__init__(batch_size=batch_size,
                                            state_size=4,
                                            measurement_size=2,
                                            log_cholesky=log_cholesky)

    def motion_model(self, states: torch.Tensor, ctrl: torch.Tensor) -> torch.Tensor:
        """
        Apply motion model to batches of sigma points

        Applies motion model to b batches of n sigma points, where each sigma point (state)
        has dimensionality m (typically: n = 2 * m + 1).

        Args:
            states: sigma points as (b, m, n) tensors
            ctrl: control-input (ignored)

        Returns:
            Advanced states
        """
        x = states[:, 0]
        y = states[:, 1]
        vx = states[:, 2]
        vy = states[:, 3]

        update = torch.empty_like(states)
        update[:, 0] = x + vx
        update[:, 1] = y + vy
        update[:, 2] = vx
        update[:, 3] = vy

        return update

    def measurement_model(self, xs: torch.Tensor) -> torch.Tensor:
        """
        Apply measurement model to batches of sigma points

        Applies measurement model to b batches of n sigma points, where each sigma point (state)
        has dimensionality m (typically: n = 2 * m + 1).

        Args:
            xs: sigma points as (b, m, n) tensors

        Returns:
            Predicted measurements
        """
        return xs[:, 0:2]


class SimpleUKF(ukf.UKF):
    def __init__(self, *args, **kwargs):
        super(SimpleUKF, self).__init__(cell=SimpleUKFCell(*args, **kwargs))

    def forward(self,
                measurements: torch.Tensor,
                state: torch.Tensor,
                state_cov: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctrl = torch.tensor(0)
        return super(SimpleUKF, self).forward(measurements, state, state_cov, ctrl)

    @property
    def process_noise(self) -> torch.tensor:
        return self.cell.process_noise.data

    @process_noise.setter
    def process_noise(self, data: torch.Tensor) -> None:
        self.cell.process_noise.data = data

    def process_noise_cov(self) -> torch.Tensor:
        return self.cell.process_noise_cov()

    @property
    def measurement_noise(self) -> torch.Tensor:
        return self.cell.measurement_noise.data

    @measurement_noise.setter
    def measurement_noise(self, data: torch.Tensor) -> None:
        self.cell.measurement_noise.data = data

    def measurement_noise_cov(self) -> torch.Tensor:
        return self.cell.measurement_noise_cov()


def init_ukf(*, batch_size: int, debug: bool = True) -> torch.Tensor:
    def _constrain_process_noise(grad):
        """
        Constrain process noise

        Constrains:
         (1)     (0, 0) and (1, 1): common average
         (2)     (2, 2) and (3, 3): common average
         (3) off-diagonal elements: zero

        Index mapping of 4x4 triangular matrix:
        [0 1 2 3 4 5 6 7 8 9] -> [[0 - - -],
                                  [1 2 - -],
                                  [3 4 5 -],
                                  [6 7 8 9]]
        """
        avg_pos = torch.sum(grad[:, (0, 2)], dim=1) / 2.
        avg_vel = torch.sum(grad[:, (5, 9)], dim=1) / 2.
        new_grad = torch.zeros_like(grad)  # (3)
        new_grad[:, 0] = avg_pos  # (1)
        new_grad[:, 2] = avg_pos  # (1)
        new_grad[:, 5] = avg_vel  # (2)
        new_grad[:, 9] = avg_vel  # (2)

        return new_grad

    def _constrain_measurement_noise(grad):
        """
        Constrain measurement noise

        Constrains:
         (1)     (0, 0) and (1, 1): common average
         (2) off-diagonal elements: zero

        Index mapping of 2x2 triangular matrix:
        [0 1 2] -> [[0 -],
                    [1 2]]
        """
        avg = torch.sum(grad[:, (0, 2)], dim=1) / 2.

        new_grad = torch.zeros_like(grad)  # (2)
        new_grad[:, 0] = avg  # (1)
        new_grad[:, 2] = avg  # (1)

        return new_grad

    def _reinit_nans(grad):
        sel = ~torch.isfinite(grad)
        if torch.any(sel):
            new_grad = torch.tensor(grad)
            new_grad[sel] = torch.rand(new_grad[sel].shape)
            print('Warning! Found {torch.sum(sel)} NaN values in gradient')
            return new_grad

    rnn = SimpleUKF(batch_size=batch_size, log_cholesky=True)

    rnn.cell.process_noise.register_hook(_reinit_nans)
    rnn.cell.process_noise.register_hook(_constrain_process_noise)

    rnn.cell.measurement_noise.register_hook(_reinit_nans)
    rnn.cell.measurement_noise.register_hook(_constrain_measurement_noise)

    return rnn if debug else torch.jit.script(rnn)
