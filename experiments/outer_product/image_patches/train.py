import jax
from jax.random import PRNGKey
import jax.numpy as jnp
import jaks
import jaks.modules as nn
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm
import typer

import json
from pathlib import Path
from typing import Iterable, Union


@dataclass
class MLP2(nn.Module):
    dims: int
    hidden_scale: float = 4.0

    def modules(self):
        hidden_dims = int(round(self.dims * self.hidden_scale))

        yield "layernorm", nn.LayerNorm(self.dims)
        yield "linear1", nn.Linear(self.dims, hidden_dims)
        yield "gelu", nn.QuickGELU()
        yield "linear2", nn.Linear(hidden_dims, self.dims)


@dataclass
class PatchesOuterProduct(nn.Module):
    in_dims: int
    num_patches: int
    out_dims: int

    def modules(self):
        yield "patches1", nn.Rotate(self.in_dims, self.num_patches)
        yield "patches2", nn.Rotate(self.in_dims, self.num_patches)
        yield "reduce", nn.Rotate(self.num_patches ** 2, self.out_dims)

    def forward(self, params, x):
        p1 = self.patches1(params, x)
        p2 = self.patches2(params, x)
        flat = jnp.outer(p1, p2).reshape(-1)
        return self.reduce(params, flat)


@dataclass 
class MNISTResNet(nn.Module):
    dims: int
    layers: int
    hidden_scale: float = 4.0

    def modules(self):
        mlp2 = MLP2(self.dims, self.hidden_scale)
        resnet = nn.ResNetStack(mlp2, self.layers)

        yield "prenorm", nn.LayerNorm(784)
        yield "patches", nn.Rotate(784, self.dims)
        yield "resnet", resnet
        yield "postnorm", nn.LayerNorm(self.dims)
        yield "out", nn.Linear(self.dims, 10)


@dataclass 
class MNISTPatches(nn.Module):
    dims: int
    num_patches: int
    layers: int
    hidden_scale: float = 4.0

    def modules(self):
        mlp2 = MLP2(self.dims, self.hidden_scale)
        resnet = nn.ResNetStack(mlp2, self.layers)

        yield "prenorm", nn.LayerNorm(784)
        yield "patches", PatchesOuterProduct(784, self.num_patches, self.dims)
        yield "resnet", resnet
        yield "postnorm", nn.LayerNorm(self.dims)
        yield "out", nn.Linear(self.dims, 10)


def load_dataset(batch_size: int, dataset: str = "mnist"):
    (ds_train, ds_test), ds_info = tfds.load(
        dataset,
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    def normalize_img(image, label):
        return tf.cast(image, tf.float32) / 255., label

    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(batch_size)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_test


def plot_compare(loss1, acc1, loss2, act2):
    epochs = np.arange(1, len(loss) + 1)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss1, label="Normal Patches")
    plt.plot(epochs, loss2, label="Outer Product")
    plt.gca().set_xlabel("Epochs")
    plt.gca().set_ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc1, label="Normal Patches")
    plt.plot(epochs, acc2, label="Outer Product")
    plt.gca().set_xlabel("Epochs")
    plt.gca().set_ylabel("Accuracy")

    plt.gcf().set_size_inches(9, 16)
    plt.savefig("results.png", dpi=120)


def train(
    net: nn.Module, 
    data: Tuple[Iterable], 
    epochs: int, 
    learning_rate: float,
    random_seed: Union[int, PRNGKey] = 0
    ):

    (train_data, test_data) = data
    if type(random_seed) is int:
        random_seed = PRNGKey(random_seed)

    key, params = net.init(random_seed)
    forward_fn = net.compile()

    def loss_fn(params, inputs, labels):
        preds = forward_fn(params, inputs)
        return jaks.utils.log_loss(preds, labels)

    @jax.jit
    def train_step(params, inputs, labels, lr):
        loss, grads = jax.value_and_grad(loss_fn)(params, inputs, labels)
        return loss, jaks.utils.sgd(params, grads, lr)

    @jax.jit
    def test_step(params, inputs, labels):
        preds = forward_fn(params, inputs)
        return jaks.utils.accuracy(preds, labels).mean()

    losses = [0]
    accs = [0]
    for epoch in range(epochs):
        pbar = tqdm(test_data, F"epoch {epoch} train", leave=False)
        loss = 0
        for i, (x, y) in enumerate(pbar):
            x = jnp.asarray(x).reshape(x.shape[0], -1)
            y = jnp.asarray(y).reshape(-1)
            _loss, params = train_step(params, x, y, learning_rate)
            loss += _loss
            losses[epoch] = loss / (i + 1)
            desc = F"epoch {epoch} train loss: {losses[epoch]:.06f}"
            pbar.set_description(desc)

        pbar = tqdm(test_data, F"epoch {epoch} test", leave=False)
        acc = 0
        for i, (x, y) in enumerate(pbar):
            x = jnp.asarray(x).reshape(x.shape[0], -1)
            y = jnp.asarray(y).reshape(-1)
            acc += test_step(params, x, y)
            accs[epoch] = 100 * acc / (i + 1)
            desc = F"epoch {epoch} test acc: {accs[epoch]:.02f}%"
            pbar.set_description(desc)

        if epoch == epochs - 1:
            return train_loss, test_acc



app = typer.Typer()


@app.command()
def run(config: Path):
    cfg = json.load(config.open('r'))

    data = load_dataset(cfg["batch_size"])
    net = MNISTResNet(cfg["dims"], cfg["layers"], cfg["hidden_scale"])
    loss1, acc1 = train(
        net, 
        data, 
        cfg["epochs"], 
        cfg["learning_rate"], 
        cfg["random_seed"]
    )

    net =  MNISTPatches(
        cfg["dims"], 
        cfg["patches"], 
        cfg["layers"], 
        cfg["hidden_scale"]
    )
    loss2, acc2 = train(
        net, 
        data, 
        cfg["epochs"], 
        cfg["learning_rate"], 
        cfg["random_seed"]
    )

    plot_compare(loss1, acc1, loss2, acc2)


if __name__ == "__main__":
    app()
