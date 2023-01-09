import logging
import os
from pathlib import Path

import click
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv
from model import MyAwesomeModel


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--epochs", default=1, help="number of epochs to train")
def train(lr, epochs):
    print("Training day and night")
    print(lr)

    if not os.path.isdir("checkpoints"):
        os.mkdir("checkpoints")

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train_set = torch.utils.data.ConcatDataset(
        [
            torch.utils.data.TensorDataset(
                torch.tensor(train_raw["images"], dtype=torch.float64),
                torch.tensor(train_raw["labels"], dtype=torch.int64),
            )
            for train_raw in [
                np.load("..\\..\\data\\processed\\train_{:d}.npz".format(i))
                for i in range(5)
            ]
        ]
    )
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=16)
    optimizer = torch.optim.Adam(lr=lr, params=model.parameters())
    loss_fn = torch.nn.NLLLoss()

    for epoch in range(epochs):
        print("EPOCH: {:d}".format(epoch))
        for i, Xy in enumerate(train_dataloader):
            X, y = Xy
            logits = model(X)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(loss.item())

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            },
            "checkpoints/checkpoint_{:d}".format(epoch),
        )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    train()
