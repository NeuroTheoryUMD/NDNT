import torch
import pytest

import sys
sys.path.append('../../') # to have access to NDNT

from NDNT.modules.activity_regularization import ActivityRegularization

@pytest.fixture
def layer_output():
    return torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]])


@pytest.fixture
def reg_vals():
    return {'activity': 0.1, 'nonneg': 0.2}


def test_activity_regularization_init(reg_vals):
    reg = ActivityRegularization(reg_vals)
    assert reg.reg_vals == reg_vals


def test_activity_regularization_activity(layer_output):
    reg = ActivityRegularization({})
    alpha = 0.1
    result = reg.activity(layer_output, alpha)
    expected = alpha * torch.mean(torch.sum(layer_output**2, axis=1), axis=0)
    assert torch.allclose(result, expected)


def test_activity_regularization_nonneg(layer_output):
    reg = ActivityRegularization({})
    alpha = 0.2
    result = reg.nonneg(layer_output, alpha)
    expected = alpha * torch.mean(torch.sum(torch.relu(-layer_output), axis=1), axis=0)
    assert torch.allclose(result, expected)


def test_activity_regularization_regularize(layer_output, reg_vals):
    reg = ActivityRegularization(reg_vals)
    result = reg.regularize(layer_output)
    activity_loss = reg.activity(layer_output, alpha=reg_vals['activity'])
    nonneg_loss = reg.nonneg(layer_output, alpha=reg_vals['nonneg'])
    expected = activity_loss + nonneg_loss
    assert torch.allclose(result, expected)
