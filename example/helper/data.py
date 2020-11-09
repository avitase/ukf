import numpy as np
import torch


class DataLoader:
    def __init__(self, *, phi, v0, n, n_repeat, noise=None):
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

        self._gt = torch.zeros((b, 2, n * n_repeat))
        self._gt[:, :, 1:] = torch.cumsum(vs[:, :, :-1], dim=2)

        if noise:
            self._x = torch.normal(self._gt, noise.unsqueeze(2).expand_as(self._gt))
        else:
            self._x = self._gt

    def __call__(self, *, window_size=None, randomize=False):
        if window_size is None:
            return self._x, self._gt

        b, _, n = self._x.shape
        n_windows = n - window_size + 1

        a = torch.arange(window_size).unsqueeze(0)
        b = torch.arange(n_windows).unsqueeze(1)
        idxs = a + b

        if randomize:
            idxs = idxs[torch.randperm(n_windows)]

        return self._x[:, :, idxs], self._gt[:, :, idxs]
