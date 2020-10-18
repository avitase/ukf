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

    def _transform(func, *, n=None, batch_size=None, mu=None, cov=None):
        if n is not None:
            mu = _gen_mu(n=n, batch_size=batch_size)
            cov = _gen_cov(n=n, batch_size=batch_size)

        mu2, cov2 = ukf.unscented_transform(mu, cov, func=func)

        assert mu2.ndim == 2
        assert mu2.shape == mu.shape

        assert cov2.ndim == 3
        assert cov2.shape == cov.shape

        return (mu, mu2), (cov, cov2)

    def _test_unit(*, n=None, batch_size=None, mu=None, cov=None):
        (mu, mu2), (cov, cov2) = _transform(lambda x: x,
                                            n=n,
                                            batch_size=batch_size,
                                            mu=mu,
                                            cov=cov)
        assert torch.allclose(mu, mu2)
        assert torch.allclose(cov, cov2)

    def _test_scale(*, n=None, batch_size=None, mu=None, cov=None):
        (mu, mu2), (cov, cov2) = _transform(lambda x: x * 3.,
                                            n=n,
                                            batch_size=batch_size,
                                            mu=mu,
                                            cov=cov)
        assert torch.allclose(mu * 3., mu2)
        assert torch.allclose(cov * 9., cov2)

    def _test_shift(*, n=None, batch_size=None, mu=None, cov=None):
        (mu, mu2), (cov, cov2) = _transform(lambda x: x + 3.,
                                            n=n,
                                            batch_size=batch_size,
                                            mu=mu,
                                            cov=cov)
        assert torch.allclose(mu + 3., mu2)
        assert torch.allclose(cov, cov2)

    mu = torch.from_numpy(np.array([[0., 0.], ]))
    cov = torch.from_numpy(np.array([[[1., 0.], [0., 1.]], ]))
    _test_unit(mu=mu, cov=cov)
    _test_scale(mu=mu, cov=cov)
    _test_shift(mu=mu, cov=cov)

    def _run(*, n, batch_size):
        n_iter = 100
        for _ in range(n_iter):
            _test_unit(n=n, batch_size=batch_size)
            _test_scale(n=n, batch_size=batch_size)
            _test_shift(n=n, batch_size=batch_size)

    for b in [1, 2, 3, 5, 100]:
        for n in [2, 3, 5, 10]:
            _run(n=n, batch_size=b)
