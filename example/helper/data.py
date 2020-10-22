import numpy as np
import torch


def generate_data(*, phi, v0, noise, n, n_repeat):
    tau = n // 3

    def _rot(x):
        c = torch.cos(x)
        s = torch.sin(x)

        b, = x.shape
        r = torch.empty((b, 2, 2))
        r[:, 0, 0] = c
        r[:, 0, 1] = -s
        r[:, 1, 0] = s
        r[:, 1, 1] = c
        return r

    b, = phi.shape
    vs = torch.empty((b, 2, 3 * tau * n_repeat))

    v = (v0.unsqueeze(1) * torch.tensor([[1., 0.], ])).unsqueeze(2)
    v = torch.matmul(_rot(phi), v)

    rot = _rot(torch.ones_like(phi) * 2. / 3. * np.pi)

    for i in range(n_repeat):
        offset = i * 3 * tau

        for j in range(3):
            begin = offset + j * tau
            end = offset + (j + 1) * tau
            vs[:, :, begin:end] = v
            v = torch.matmul(rot, v)

    gt = torch.cumsum(vs, axis=2)
    x = torch.normal(gt, noise.unsqueeze(2).expand_as(gt))

    return x, gt
