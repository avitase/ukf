import torch
from ukf import ukf_step


def test_ukf():
    def _motion_model(x):
        dt = .5
        a = -2.

        x_updated = torch.empty_like(x)
        x_updated[:, 0] = x[:, 0] + x[:, 1] * dt
        x_updated[:, 1] = x[:, 1] + a * dt

        return x_updated

    def _measurement_model(x):
        s = 20.
        d = 40.

        return torch.atan(s / (d - x[:, 0:1, :]))

    state = torch.tensor([[0., 5.], ])
    state_cov = torch.tensor([[[.01, 0.], [0., 1.]], ])
    process_noise = torch.tensor([[[.1, 0.], [0., .1]], ])
    measurement_noise = torch.tensor([[[.01, ]], ])

    x, y, cov_x, cov_y, gain = ukf_step(motion_model=_motion_model,
                                        measurement_model=_measurement_model,
                                        state=state,
                                        state_cov=state_cov,
                                        process_noise=process_noise,
                                        measurement_noise=measurement_noise)

    x_exp = torch.tensor([[2.5, 4.], ])
    y_exp = torch.tensor([[.49], ])
    cov_x_exp = torch.tensor([[[.36, .5], [.5, 1.1]], ])
    cov_y_exp = torch.tensor([[[.01, ]], ])
    gain_exp = torch.tensor([[[.4, ], [.55, ]], ])

    assert torch.allclose(x, x_exp, atol=1e-04)
    assert torch.allclose(y, y_exp, atol=1e-04)
    assert torch.allclose(cov_x, cov_x_exp, atol=1e-04)
    assert torch.allclose(cov_y, cov_y_exp, atol=1e-04)
    assert torch.allclose(gain, gain_exp, atol=1e-02)
