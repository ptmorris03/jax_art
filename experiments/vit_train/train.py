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

import numpy as np
import jax
import jax.numpy as jnp
from jax.nn.initializers import variance_scaling
import jaks
import jaks.modules as nn

from pathlib import Path

import matplotlib.pyplot as plt
from einops import rearrange, reduce
import optax

from dataclasses import dataclass
from collections import OrderedDict
from functools import partial
from typing import Iterable, Optional, Union, Tuple, Callable

from sklearn.decomposition import PCA
from matplotlib import font_manager
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from imageio import imwrite

from math import sqrt
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Tuple, Union
import pickle

from datasets import mnist


def zscore(x, eps=1e-5):
    return (x - x.mean()) / (x.std() + eps)


@dataclass
class LayerNorm(nn.Module):
    dims: int

    def parameters(self, random_key):
        yield "scale", jnp.ones(self.dims)
        yield "translate", jnp.zeros(self.dims)

    def forward(self, params, x):
        return zscore(x) * params["scale"] + params["translate"]


@dataclass
class Linear(nn.Module):
    in_dims: int
    out_dims: int

    def parameters(self, random_key):
        matrix = jax.random.normal(random_key, (self.out_dims, self.in_dims)) * sqrt(2 / self.in_dims)
        yield "weight", matrix
        yield "bias", jnp.zeros((self.out_dims,))

    def forward(self, params, x):
        return jnp.matmul(params["weight"], x) + params["bias"]


@dataclass
class Patches(nn.Module):
    patches: int
    channels: int
    patch_shape: Tuple[int]
    image_shape: Tuple[int]
    position: bool = True

    def parameters(self, key):
        next_key = jaks.utils.PRNGSplitter(key)
        
        shape = (self.patches, self.channels, *self.patch_shape)
        dims = self.channels
        for dim in self.patch_shape:
            dims *= dim
        yield "filters", jax.random.normal(next_key(), shape) * sqrt(2 / dims)

        if self.position:
            dim = 1
            for pdim, idim in zip(self.patch_shape, self.image_shape):
                dim *= idim // pdim
            shape = (dim, self.patches)
            dims = self.patches
            yield "position", jax.random.normal(next_key(), shape) * sqrt(2 / dims)

    def forward(self, params, x):
        n_dims = len(self.patch_shape)
        dims = ' '.join(['dim' + str(i + 1) for i in range(n_dims)])

        x = jax.lax.conv(x, params["filters"], self.patch_shape, padding='SAME')
        x = rearrange(x, F"batch channel {dims} -> batch ({dims}) channel")
        if self.position: 
            x = x + params["position"]
        return x


@dataclass
class SelfAttention(nn.Module):
    dims: int
    heads: int

    def modules(self):
        yield "layernorm", nn.Vmap(LayerNorm(self.dims))
        yield "qkv", nn.Vmap(Linear(self.dims, self.dims * 3))
        yield "out", nn.Vmap(Linear(self.dims, self.dims))

    def forward(self, params, x):
        x = self.layernorm(params, x)
        qkv = self.qkv(params, x)
        qkv = rearrange(qkv, "patch (d2 head d) -> d2 head patch d", d2=3, head=self.heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = rearrange(k, "head patch d -> head d patch")
        attn = jax.nn.softmax(jax.vmap(jnp.matmul)(q, k_t) / (k.shape[-1] ** 0.5), axis=-1)
        values = jax.vmap(jnp.matmul)(attn, v)
        values = rearrange(values, "head patch d -> patch (head d)")
        return self.out(params, values)


@dataclass
class Lambda(nn.Module):
    function: Callable

    def forward(self, params, x):
        return self.function(x)


@dataclass
class MLP2(nn.Module):
    dims: int
    hidden_scale: float = 4

    def modules(self):
        hidden_dims = int(round(self.dims * self.hidden_scale))

        yield "layernorm", LayerNorm(self.dims)
        yield "linear1", Linear(self.dims, hidden_dims)
        yield "activation", Lambda(jax.nn.relu)
        yield "linear2", Linear(hidden_dims, self.dims)


@dataclass
class EncoderBlock(nn.Module):
    dims: int
    heads: int
    hidden_scale: float = 4

    def modules(self):
        yield "attn", SelfAttention(self.dims, self.heads)
        yield "mlp2", nn.Vmap(MLP2(self.dims, self.hidden_scale))

    def forward(self, params, x):
        x = x + self.attn(params, x)
        return x + self.mlp2(params, x)


class ParallelEncoderBlock(EncoderBlock):
    def forward(self, params, x):
        return x + self.attn(params, x) + self.mlp2(params, x)


@dataclass
class PoolPatches(nn.Module):
    method = 'mean'

    def forward(self, params, x):
        return reduce(x, "patch dim -> dim", self.method)


@dataclass
class ViT(nn.Module):
    layers: int
    dims: int
    heads: int
    classes: int
    patch_shape: Tuple[int]
    image_shape: Tuple[int]
    image_channels: int

    def modules(self):
        yield "patches", Patches(self.dims, self.image_channels, self.patch_shape, self.image_shape)
        for i in range(self.layers):
            yield F"encoder{i+1}", nn.Vmap(EncoderBlock(self.dims, self.heads))
        yield "pool_patches", nn.Vmap(PoolPatches())
        yield "cls_head", nn.Vmap(Linear(self.dims, self.classes))


def load_dataset(batch_size: int):
    train_images, train_labels, test_images, test_labels = mnist(permute_train=True)
    def train_data():
        for i in range(0, train_images.shape[0], batch_size):
            yield train_images[i:i+batch_size], train_labels[i:i+batch_size]
    def test_data():
        for i in range(0, test_images.shape[0], batch_size):
            yield test_images[i:i+batch_size], test_labels[i:i+batch_size]

    return train_data, test_data


def save(path, params, step):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with Path(path, F"weights_{step:06}.pickle").open('wb') as f:
        pickle.dump(params, f)


def train(
    layers = 6,
    dims = 16,
    heads =  4,
    patch_size =  4,
    seed = 4,
    batch_size =  128,
    accum_steps =  1,
    learning_rate =  1e-2,
    w_decay = 5e-4,
    epochs = 100
    ):


    vit = ViT(layers, dims, heads, 10, (patch_size, patch_size), (28, 28), 1)

    key, params = vit.init(seed, params={})
    forward_fn = vit.compile(batch=False)

    (train_data, test_data) = load_dataset(batch_size)

    opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.lamb(learning_rate=learning_rate, weight_decay=w_decay)
    )
    opt = optax.MultiSteps(opt, accum_steps, use_grad_mean=False)

    def loss_fn(params, inputs, labels):
        preds = forward_fn(params, inputs)
        return jaks.utils.log_loss(preds, labels)

    def grad_norm(grads):
        grads = jax.tree_map(lambda x: x.reshape(-1), grads)
        grad_arrays, _ = jax.tree_flatten(grads)
        return jnp.linalg.norm(jnp.concatenate(grad_arrays))

    @partial(jax.pmap, axis_name="device_batch", in_axes=(None, 0, 0, None), out_axes=(None, None, None, None))
    @jax.jit
    def train_step(params, inputs, labels, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params, inputs, labels)
        updates, opt_state = opt.update(grads, opt_state, params)
        updates = jax.lax.pmean(updates, 'device_batch')
        params = optax.apply_updates(params, updates)
        return loss, params, opt_state, grad_norm(grads)

    @partial(jax.pmap, in_axes=(None, 0, 0), out_axes=(None,))
    @jax.jit
    def test_step(params, inputs, labels):
        preds = forward_fn(params, inputs)
        return jaks.utils.accuracy(preds, labels).mean()

    opt_state = opt.init(params)
    for epoch in range(1, epochs + 1):
        pbar = tqdm(train_data(), F"epoch {epoch} train")
        loss = 0
        for i, (x, y) in enumerate(pbar):
            x = jnp.asarray(x).reshape(2, x.shape[0] // 2, 1, 28, 28)
            sign = np.random.choice((-1, 1))
            a = np.random.rand()
            x = a * x + (1 - a) * np.roll(x, (sign, sign), axis=(-1, -2))
            y = jnp.asarray(y).reshape(2, -1)
            _loss, params, opt_state, gn = train_step(params, x, y, opt_state)
            loss += _loss
            pbar.set_description(F"epoch {epoch} train_loss: {loss/(i+1):.06f} grad_norm: {gn:.06f}")
        
        pbar = tqdm(test_data(), F"epoch {epoch} test")
        acc = 0
        for i, (x, y) in enumerate(pbar):
            x = jnp.asarray(x).reshape(x.shape[0], 1, 28, 28)
            y = jnp.asarray(y).reshape(-1)
            acc += test_step(params, x, y)
            pbar.set_description(F"epoch {epoch} test_acc: {100*acc/(i+1):.02f}%")


app = typer.Typer()


@app.command()
def run(config: Path):
    cfg = json.load(config.open('r'))

    train(
        cfg["layers"],
        cfg["dims"],
        cfg["heads"],
        cfg["patch_size"],
        cfg["seed"],
        cfg["batch_size"],
        cfg["accum_steps"],
        cfg["learning_rate"],
        cfg["w_decay"],
        cfg["epochs"]
    )


if __name__ == "__main__":
    app()
