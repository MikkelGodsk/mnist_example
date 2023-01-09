import os

import numpy as np
import torch


def mnist(path='data', batch_size=16):
    train_set = torch.utils.data.ConcatDataset(
        [
            torch.utils.data.TensorDataset(
                torch.tensor(train_raw["images"], dtype=torch.float64),
                torch.tensor(train_raw["labels"], dtype=torch.int64),
            )
            for train_raw in [
                np.load(os.path.join(path, "processed\\train_{:d}.npz".format(i))) for i in range(8)
            ]
        ]
    )
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    test_raw = np.load(os.path.join(path, "processed\\test.npz"))
    test_set = torch.utils.data.TensorDataset(
        torch.tensor(test_raw["images"], dtype=torch.float64),
        torch.tensor(test_raw["labels"], dtype=torch.int64),
    )
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

    return train_dataloader, test_dataloader
