import torch
import numpy as np
import pytest

from src.data.make_dataset import mnist
from src.models.model import MyAwesomeModel


def test_model_shape():
    model = MyAwesomeModel()
    train, _ = mnist(batch_size=16)

    X, y = next(iter(train))
    y_hat = model(X)
    assert y.shape[0] == y_hat.shape[0], "Model did not produce the correct output shape"


@pytest.mark.parametrize(["shape"], [([16,1,28,28],), ([28,28],)])
def test_model_wrong_shape(shape):
    model = MyAwesomeModel()
    with pytest.raises(ValueError, match="Input should be in 3 dimensions"):
        X = torch.zeros(shape)
        model(X)