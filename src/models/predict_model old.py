import logging
from pathlib import Path

import click
import numpy as np
import torch
import torchmetrics
from dotenv import find_dotenv, load_dotenv
from model import MyAwesomeModel


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    checkpoint = torch.load(model_checkpoint)
    model = MyAwesomeModel()
    model.load_state_dict(checkpoint["model_state_dict"])

    test_raw = np.load("..\\..\\data\\processed\\test.npz")
    test_set = torch.utils.data.TensorDataset(
        torch.tensor(test_raw["images"], dtype=torch.float64),
        torch.tensor(test_raw["labels"], dtype=torch.int64),
    )
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=16)
    predictions = []
    true = []
    for X, y in test_dataloader:
        predictions.append(model(X))
        true.append(y)
    predictions = torch.vstack(predictions)
    true = torch.hstack(true)
    top1_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
    print(top1_acc(predictions, true))


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    evaluate()
