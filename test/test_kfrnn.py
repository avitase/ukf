from typing import Tuple

import torch
import torch.nn as nn

import ukf


class MyCell(nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()

    def forward(self,
                x: torch.Tensor,
                state: torch.Tensor,
                state_cov: torch.Tensor,
                ctrl: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert ctrl.item() == 42
        return x * 2, state * 2, state_cov * 2


def test_kfrnn():
    cell = ukf.KFRNN(MyCell())
    cell = torch.jit.script(cell)

    torch.manual_seed(0)
    x = torch.randint(high=10, size=(2, 3, 6))
    init_state = torch.randint(high=10, size=(2, 4))
    init_state_cov = torch.randint(high=10, size=(2, 4, 4))

    ctrl = torch.tensor(42, dtype=torch.int)
    preds, states, state_covs = cell(x, init_state, init_state_cov, ctrl)
    assert preds.shape == (2, 3, 6)
    assert states.shape == (2, 4, 6)
    assert state_covs.shape == (2, 4, 4, 6)

    assert torch.all(preds == x * 2)
    for i in range(5):
        f = 2 ** (i + 1)
        assert torch.all(states[:, :, i] == init_state * f)
        assert torch.all(state_covs[:, :, :, i] == init_state_cov * f)
