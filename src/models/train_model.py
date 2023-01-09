import logging
import os
from pathlib import Path

import click
import hydra
import torch
import wandb
from dotenv import find_dotenv, load_dotenv
from model import MyAwesomeModel

from data.make_dataset import mnist


@click.command()
@hydra.main(config_path="../conf", config_name="config.yaml")
def train(cfg):
    wandb.init()

    print("Training day and night")

    if not os.path.isdir("checkpoints"):
        os.mkdir("checkpoints")

    # TODO: Implement training loop here
    model = MyAwesomeModel(cfg.model.hidden_neurons)
    train_dataloader, _ = mnist(batch_size=cfg.training.batch_size)
    optimizer = torch.optim.Adam(lr=cfg.training.lr, params=model.parameters())
    loss_fn = torch.nn.NLLLoss()

    model(next(iter(train_dataloader))[0])  # Initialize weights and biases of model
    wandb.watch(model, log_freq=cfg.training.log_interval)

    for epoch in range(cfg.training.epochs):
        print("EPOCH: {:d}".format(epoch))
        for i, Xy in enumerate(train_dataloader):
            X, y = Xy
            logits = model(X)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % cfg.training.log_interval == 0:
                example = wandb.Image(X[0])
                wandb.log({"loss": loss, "example": example, "step": i})

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
