import numpy as np
import torch
import torch.optim

import ukf


class MyUKFCell(ukf.UKFCell):
    def __init__(self, *args, **kwargs):
        super(MyUKFCell, self).__init__(*args, **kwargs)

    def motion_model(self, x):
        dt = .5
        a = -2.

        x_updated = torch.empty_like(x)
        x_updated[:, 0] = x[:, 0] + x[:, 1] * dt
        x_updated[:, 1] = x[:, 1] + a * dt

        return x_updated

    def measurement_model(self, x):
        s = 20.
        d = 40.

        return torch.atan(s / (d - x[:, 0:1, :]))


def test_ukfcell():
    cell = MyUKFCell(batch_size=2, state_size=2, measurement_size=1)
    cell.process_noise.data = torch.sqrt(torch.tensor([[.1, 0., .1], [.2, 0., 3.]]))
    cell.measurement_noise.data = torch.sqrt(torch.tensor([[.01, ], [.04, ]]))
    cell = torch.jit.script(cell)

    init_state = torch.tensor([[0., 5.], [1., 6.]])
    init_state_cov = torch.tensor([
        [[.01, 0.], [0., 1.]],
        [[.09, 0.], [0., 4.]],
    ])

    phi = torch.tensor([[np.pi / 6., ], [np.pi / 3., ]])
    with torch.no_grad():
        y, new_state, new_cov = cell(phi, init_state, init_state_cov)

    y_exp = torch.tensor([[.49], [.5074]])
    x_exp = torch.tensor([[2.5133, 4.0185], [4.2047, 5.3173]])
    x_cov_exp = torch.tensor([
        [[0.3584, 0.4978], [0.4978, 1.0969]],
        [[1.2842, 1.9910], [1.9910, 6.9861]],
    ])

    assert torch.allclose(y, y_exp, atol=1e-04)
    assert torch.allclose(new_state, x_exp, atol=1e-04)
    assert torch.allclose(new_cov, x_cov_exp, atol=1e-04)
