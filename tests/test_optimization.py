import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
from pytest import approx

from pyriemann.geometry.distance import distance
from pyriemann.optimization.grassmann import _grad, _loss


@pytest.mark.parametrize("metric", ["euclid", "riemann"])
def test_grassmann_loss(metric, get_mats, get_weights):
    """Test that loss is the weighted sum of squared distances"""
    n_matrices, n_channels = 3, 4
    X = get_mats(n_matrices, n_channels, "spd")
    Y = get_mats(n_matrices, n_channels, "spd")
    weights = get_weights(n_matrices)
    Q = get_mats(1, n_channels, "orth")[0]

    assert _loss(Q, X, Y, weights, metric=metric) == approx(
        weights @ distance(X, Q @ Y @ Q.T, metric=metric) ** 2
    )


@pytest.mark.parametrize("metric", ["euclid", "riemann"])
def test_grassmann_grad(metric, get_mats, get_weights):
    """Test that gradient is the derivative of the loss"""
    n_matrices, n_channels = 3, 4
    X = get_mats(n_matrices, n_channels, "spd")
    Y = get_mats(n_matrices, n_channels, "spd")
    weights = get_weights(n_matrices)
    Q = get_mats(1, n_channels, "orth")[0]

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
