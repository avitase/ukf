import numpy as np
import torch

from ukf import ukf_step, kf_correct


class UKF:
    def __init__(self, *, phi_noise):
        self.b, = phi_noise.shape
        self.n = 3
        self.m = 2

        self.state = torch.zeros(self.n).repeat(self.b, 1)
        self.state_cov = torch.eye(self.n).repeat(self.b, 1, 1)
        self.phi_noise = phi_noise
        self.measurement_noise = (torch.eye(self.m) * .5).repeat(self.b, 1, 1)

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
        process_noise = torch.zeros(self.b, self.n, self.n)
        process_noise[:, 2, 2] = self.phi_noise

        x, y, cov_x, cov_y, gain = ukf_step(motion_model=UKF.motion_model,
                                            measurement_model=UKF.measurement_model,
                                            state=self.state,
                                            state_cov=self.state_cov,
                                            process_noise=process_noise,
                                            measurement_noise=self.measurement_noise)
        self.state, self.state_cov = kf_correct(measurement,
                                                x=x,
                                                y_predicted=y,
                                                cov_x=cov_x,
                                                cov_y=cov_y,
                                                gain=gain)
        error = measurement - y
        return x, torch.sum(error ** 2, dim=1)
