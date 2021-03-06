import warnings

import numpy as np
import torch
import torch.optim

import ukf


class MyUKFCell(ukf.UKFCell):
    def __init__(self, *args, **kwargs):
        super(MyUKFCell, self).__init__(*args, **kwargs)

    def motion_model(self, x, ctrl):
        assert ctrl.item() == 42

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


def jit_compile(obj, debug=False):
    if debug:
        warnings.warn('JIT compilation is skipped!')
        return obj

    return torch.jit.script(obj)


def test_ukfcell():
    cell = MyUKFCell(batch_size=2, state_size=2, measurement_size=1, log_cholesky=False)
    cell.process_noise.data = torch.sqrt(
        torch.tensor([[.1, 0., .1], [.2, 0., 3.]], dtype=torch.double))
    cell.measurement_noise.data = torch.sqrt(
        torch.tensor([[.01, ], [.04, ]], dtype=torch.double))
    cell = jit_compile(cell)

    init_state = torch.tensor([[0., 5.], [1., 6.]], dtype=torch.double)
    init_state_cov = torch.tensor([
        [[.01, 0.], [0., 1.]],
        [[.09, 0.], [0., 4.]],
    ], dtype=torch.double)

    phi = torch.tensor([[np.pi / 6., ], [np.pi / 3., ]], dtype=torch.double)
    with torch.no_grad():
        ctrl = torch.tensor(42, dtype=torch.int)
        y, new_state, new_cov = cell(phi, init_state, init_state_cov, ctrl)

    y_exp = torch.tensor([[.49], [.5074]], dtype=torch.double)
    x_exp = torch.tensor([[2.5133, 4.0185], [4.2047, 5.3173]], dtype=torch.double)
    x_cov_exp = torch.tensor([
        [[0.3584, 0.4978], [0.4978, 1.0969]],
        [[1.2842, 1.9910], [1.9910, 6.9861]],
    ], dtype=torch.double)

    assert torch.allclose(y, y_exp, atol=1e-04)
    assert torch.allclose(new_state, x_exp, atol=1e-04)
    assert torch.allclose(new_cov, x_cov_exp, atol=1e-04)


def test_ukfcell_log_cholesky():
    def _log_sqrt(x):
        return torch.log(torch.sqrt(torch.tensor(x)))

    cell = MyUKFCell(batch_size=2, state_size=2, measurement_size=1, log_cholesky=True)
    cell.process_noise.data = torch.tensor([
        [_log_sqrt(.1), 0., _log_sqrt(.1)],
        [_log_sqrt(.2), 0., _log_sqrt(3.)],
    ], dtype=torch.double)
    cell.measurement_noise.data = torch.tensor([
        [_log_sqrt(.01), ], [_log_sqrt(.04), ],
    ], dtype=torch.double)
    cell = jit_compile(cell)

    init_state = torch.tensor([[0., 5.], [1., 6.]], dtype=torch.double)
    init_state_cov = torch.tensor([
        [[.01, 0.], [0., 1.]],
        [[.09, 0.], [0., 4.]],
    ], dtype=torch.double)

    phi = torch.tensor([[np.pi / 6., ], [np.pi / 3., ]], dtype=torch.double)
    with torch.no_grad():
        ctrl = torch.tensor(42, dtype=torch.int)
        y, new_state, new_cov = cell(phi, init_state, init_state_cov, ctrl)

    y_exp = torch.tensor([[.49], [.5074]], dtype=torch.double)
    x_exp = torch.tensor([[2.5133, 4.0185], [4.2047, 5.3173]], dtype=torch.double)
    x_cov_exp = torch.tensor([
        [[0.3584, 0.4978], [0.4978, 1.0969]],
        [[1.2842, 1.9910], [1.9910, 6.9861]],
    ], dtype=torch.double)

    assert torch.allclose(y, y_exp, atol=1e-04)
    assert torch.allclose(new_state, x_exp, atol=1e-04)
    assert torch.allclose(new_cov, x_cov_exp, atol=1e-04)
