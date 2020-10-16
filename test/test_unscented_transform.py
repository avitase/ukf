import numpy as np
import torch

import ukf


def test_unscented_transform():
    rs = np.random.RandomState(42)

    def _gen_mu(*, n, batch_size):
        return torch.from_numpy(rs.random((batch_size, n)))

    def _gen_cov(*, n, batch_size):
        L = torch.from_numpy(rs.random((batch_size, n, n)))
        return torch.matmul(L, L.transpose(1, 2))

    def _test_unit(*, n=None, batch_size=None, mu=None, cov=None):
        if n is not None:
            mu = _gen_mu(n=n, batch_size=batch_size)
            cov = _gen_cov(n=n, batch_size=batch_size)

        mu2, cov2 = ukf.unscented_transform(mu, cov, func=lambda x: x)

        assert mu2.ndim == 2
        assert mu2.shape == mu.shape

        assert cov2.ndim == 3
        assert cov2.shape == cov.shape

        assert torch.allclose(mu, mu2)
        assert torch.allclose(cov, cov2)

    mu = torch.from_numpy(np.array([[0., 0.], ]))
    cov = torch.from_numpy(np.array([[[1., 0.], [0., 1.]], ]))
    _test_unit(mu=mu, cov=cov)

    def _run(*, batch_size):
        n_iter = 100
        for _ in range(n_iter):
            _test_unit(n=2, batch_size=batch_size)
            _test_unit(n=3, batch_size=batch_size)
            _test_unit(n=5, batch_size=batch_size)
            _test_unit(n=10, batch_size=batch_size)

    _run(batch_size=1)
    _run(batch_size=2)
    _run(batch_size=3)
    _run(batch_size=5)
    _run(batch_size=100)
