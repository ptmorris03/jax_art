import jax
import jax.numpy as jnp
import jaks
import jaks.modules as nn
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm
import typer


app = typer.Typer()


@app.command()
def train(epochs: int, batch_size: int):
    pass


if __name__ == "__main__":
    app()
