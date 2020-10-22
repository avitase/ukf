import numpy as np
import torch

from ukf import ukf_step, ukf_correct


class UKF:
    def __init__(self, *, b):
        self.state = torch.zeros(3).repeat(b, 1)
        self.state_cov = torch.eye(3).repeat(b, 1, 1)
        self.process_noise = torch.tensor([
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., np.pi / 6.],
        ]).repeat(b, 1, 1)
        self.measurement_noise = (torch.eye(2) * .5).repeat(b, 1, 1)

    @staticmethod
    def motion_model(state):
        x = state[:, 0]
        y = state[:, 1]
        phi = state[:, 2]
        v = torch.tensor([1., 2., 3., 4.]).unsqueeze(1)

        # apply same correction to all sigma points
        phi_mean = phi[:, 0]
        phi[phi_mean > np.pi] -= 2. * np.pi
        phi[phi_mean < -np.pi] += 2. * np.pi

        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)

        vx = v * cos_phi
        vy = v * sin_phi

        update = torch.empty_like(state)
        update[:, 0] = x + vx
        update[:, 1] = y + vy
        update[:, 2] = phi

        return update

    @staticmethod
    def measurement_model(x):
        return x[:, 0:2]

    def step(self, measurement):
        x, y, cov_x, cov_y, gain = ukf_step(motion_model=UKF.motion_model,
                                            measurement_model=UKF.measurement_model,
                                            state=self.state,
                                            state_cov=self.state_cov,
                                            process_noise=self.process_noise,
                                            measurement_noise=self.measurement_noise)
        self.state, self.state_cov = ukf_correct(measurement,
                                                 x=x,
                                                 y_predicted=y,
                                                 cov_x=cov_x,
                                                 cov_y=cov_y,
                                                 gain=gain)
        return self.state, measurement - y
