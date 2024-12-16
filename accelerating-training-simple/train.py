from src.classifier import Classifier
from src.vgg import vgg19
from src.utils import count_parameters, load_weights
from src.train_utils import train
from src.dataloader import load_cifar
from src.decomposition import tucker_decompose_model

import os
import click
import wandb

import tensorly as tl

@click.command()
@click.option("--name", default="accelerating_training", help="Name.")
@click.option("--model", default="simple", help="Model.")
@click.option("--device", default="cuda:0", help="Device.")
@click.option("--num_epoch", default=300, help="Number of training epoch.")
@click.option("--checkpoint", default=None, help="Checkpoint path.")
@click.option("--linear_rank", default=None, help="Model ranks for linear.")
@click.option("--conv_rank", default=None, help="Model ranks for conv.")
@click.option("--batch_size", default=64, help="Batch size.")
def fit(name, model, device, num_epoch, checkpoint, linear_rank, conv_rank, batch_size):
    run = wandb.init(project="accelerating_training", name=name)
    run.config.model = model
    run.config.device = device
    run.config.num_epoch = num_epoch
    run.config.checkpoint = checkpoint
    run.config.linear_rank = linear_rank
    run.config.conv_rank = conv_rank
    run.config.batch_size = batch_size

    path = os.path.join(f"out/{name}")
    if not os.path.exists(path):
        os.makedirs(path)

    train_loader, eval_loader = load_cifar(train_batch_size=batch_size, eval_batch_size=batch_size)
    if model == "simple":
        model = Classifier()
    elif model == "large":
        model = vgg19()
    else:
        raise ValueError("No such model.")

    if checkpoint is not None:
        load_weights(model, checkpoint)

    if linear_rank is not None or conv_rank is not None:
        tl.set_backend("pytorch")
        model = tucker_decompose_model(model, linear_max_rank=int(linear_rank), conv_max_rank=int(conv_rank))

    print(model)

    num_params, top = count_parameters(model, return_top=True)
    print("num params:", num_params, "top:", top)

    train(
        model=model,
        device=device,
        path=path,
        run=run,
        train_loader=train_loader,
        eval_loader=eval_loader,
        num_epoch=num_epoch
    )

if __name__ == "__main__":
    fit()