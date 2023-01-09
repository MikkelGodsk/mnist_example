import torch
import numpy as np
import pytest

from src.data.make_dataset import mnist
from src.models.model import MyAwesomeModel


def test_model_shape():
    model = MyAwesomeModel()

    X = torch.randn([16,28,28], dtype=torch.float64)
    y_hat = model(X)
    assert y_hat.shape == torch.Size([16,10]), "Model did not produce the correct output shape"


@pytest.mark.parametrize(["shape"], [([16,1,28,28],), ([28,28],)])
def test_model_wrong_shape(shape):
    model = MyAwesomeModel()
    with pytest.raises(ValueError, match="Input should be in 3 dimensions"):
        X = torch.zeros(shape)
        model(X)