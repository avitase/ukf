import torch

import ukf


class SimpleUKFCell(ukf.UKFCell):
    def __init__(self, *, batch_size: int):
        super(SimpleUKFCell, self).__init__(batch_size=batch_size, state_size=4, measurement_size=2)

    def motion_model(self, states: torch.Tensor) -> torch.Tensor:
        """
        Applies motion model to batches of sigma points

        Applies motion model to b batches of n sigma points, where each sigma point (state)
        has dimensionality m (typically: n = 2 * m + 1).

        Args:
            states: sigma points as (b, m, n) tensors

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
        Applies measurement model to batches of sigma points

        Applies measurement model to b batches of n sigma points, where each sigma point (state)
        has dimensionality m (typically: n = 2 * m + 1).

        Args:
            xs: sigma points as (b, m, n) tensors

        Returns:
            Predicted measurements
        """
        return xs[:, 0:2]


class SimpleUKFRNN(ukf.KFRNN):
    def __init__(self, *args, **kwargs):
        super(SimpleUKFRNN, self).__init__(cell=SimpleUKFCell(*args, **kwargs))
