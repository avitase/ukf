from typing import Tuple

import torch
import torch.nn as nn


class KFRNN(nn.Module):
    def __init__(self, cell):
        super(KFRNN, self).__init__()
        self.cell = cell

    def forward(self,
                measurements: torch.Tensor,
                state: torch.Tensor,
                state_cov: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Applies forward method of cell to passed measurements successively

        Args:
            measurements: batched measurements as (b, m, s) tensor
            state: batched initial state as (b, n) tensor
            state_cov: batched initial state covariance as (b, n, n) tensor

        Returns:
            Batched predictions, states and state covariances of each step

        """
        _, _, s = measurements.shape
        b, n = state.shape

        preds = torch.empty_like(measurements)
        states = torch.empty((b, n, s + 1))
        state_covs = torch.empty((b, n, n, s + 1))

        states[:, :, 0] = state
        state_covs[:, :, :, 0] = state_cov

        for i in range(s):
            preds[:, :, i], states[:, :, i + 1], state_covs[:, :, :, i + 1] = self.cell.forward(
                measurements[:, :, i], states[:, :, i], state_covs[:, :, :, i]
            )

        return preds, states[:, :, 1:], state_covs[:, :, :, 1:]
