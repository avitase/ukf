import numpy as np
import torch

import ukf

if __name__ == '__main__':
    # TODO: test for n > 3 (kappa = 0 for n = 3)
    mu = torch.from_numpy(np.array([
        [1, 2, 3],
        [4, 5, 6],
    ], dtype=np.float32))
    cov = torch.from_numpy(np.array([
        [[4, 12, -16], [12, 37, -43], [-16, -43, 98]],
        [[2, -1, 0], [-1, 2, -1], [0, -1, 2]],
    ], dtype=np.float32))
    mu2, cov2 = ukf.unscented_transform(mu, cov, func=lambda x: x)
    print(mu2)
    print(cov2)
