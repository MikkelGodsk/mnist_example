import os

import numpy as np
import pytest
import torch

from src.data.make_dataset import mnist
from tests import _PATH_DATA


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_data_length():
    N_train = 40000  # 25000
    N_test = 5000

    train, test = mnist(batch_size=16)
    len_train = 0
    for X, y in train:
        len_train += len(y)
    len_test = 0
    for X, y in test:
        len_test += len(y)
    assert len_train == N_train, "Training set had wrong length"
    assert len_test == N_test, "Test set had wrong length"


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_data_shape():
    N_train = 40000  # 25000
    N_test = 5000

    train, test = mnist(batch_size=16)
    X_train = next(iter(train))
    X_test = next(iter(test))
    assert X_train[0].shape == (16, 28, 28), "Training set sample had incorrect shape"
    assert X_test[0].shape == (16, 28, 28), "Test set sample had incorrect shape"


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_data_labels_count():
    N_train = 40000  # 25000
    N_test = 5000

    train, test = mnist(batch_size=16)
    labels_count = np.zeros((10,))
    for X, y in train:
        for label in y:
            labels_count[label] += 1
    assert np.all(labels_count), "Not all classes are present in the training set"

    labels_count = np.zeros((10,))
    for X, y in test:
        for label in y:
            labels_count[label] += 1
    assert np.all(labels_count), "Not all classes are present in the test set"
