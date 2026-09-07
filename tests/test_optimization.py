import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
from pytest import approx

from pyriemann.datasets.simulated import make_matrices
from pyriemann.geometry.distance import distance
from pyriemann.optimization.grassmann import _grad, _loss


@pytest.mark.parametrize("metric", ["euclid", "riemann"])
def test_grassmann_loss(rndstate, metric):
    """Test that loss is the weighted sum of squared distances"""
    n_matrices, n_channels = 3, 4
    X = make_matrices(n_matrices, n_channels, "spd", rs=rndstate)
    Y = make_matrices(n_matrices, n_channels, "spd", rs=rndstate)
    weights = np.array([0.5, 0.3, 0.2])
    Q = np.linalg.qr(rndstate.randn(n_channels, n_channels))[0]

    assert _loss(Q, X, Y, weights, metric=metric) == approx(
        weights @ distance(X, Q @ Y @ Q.T, metric=metric) ** 2
    )


@pytest.mark.parametrize("metric", ["euclid", "riemann"])
def test_grassmann_grad(rndstate, metric):
    """Test that gradient is the derivative of the loss"""
    n_matrices, n_channels = 3, 4
    X = make_matrices(n_matrices, n_channels, "spd", rs=rndstate)
    Y = make_matrices(n_matrices, n_channels, "spd", rs=rndstate)
    weights = np.array([0.5, 0.3, 0.2])
    Q = np.linalg.qr(rndstate.randn(n_channels, n_channels))[0]

    eps = 1e-6
    grad_num = np.zeros((n_channels, n_channels))
    for i in range(n_channels):
        for j in range(n_channels):
            step = np.zeros((n_channels, n_channels))
            step[i, j] = eps
            grad_num[i, j] = (
                _loss(Q + step, X, Y, weights, metric=metric)
                - _loss(Q - step, X, Y, weights, metric=metric)
            ) / (2 * eps)

    grad = _grad(Q, X, Y, weights, metric=metric)
    assert grad.dtype == grad_num.dtype
    assert_array_almost_equal(grad, grad_num, decimal=5)
