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

from scipy.spatial.distance import cdist
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


def load_dataset():
    return mnist(permute_train=False)


def load(path, name="weights"):
    path = Path(path)
    with Path(path, F"{name}.pickle").open('rb') as f:
        params = pickle.load(f)
        return params["params"], params["config"]


app = typer.Typer()


@app.command()
def run(weights: Path = "./"):
    

    params, cfg = load(weights)
    vit = ViT(cfg["layers"], cfg["dims"], cfg["heads"], 10, (cfg["patch_size"], cfg["patch_size"]), (28, 28), 1)
    forward_fn = vit.compile(batch=False)

    X, Y, X_test, Y_test = load_dataset()
    zero_idxs, one_idxs = np.where(Y==0)[0], np.where(Y==1)[0]
    print(zero_idxs.shape, one_idxs.shape, X.shape)
    dists = cdist(X[zero_idxs], X[one_idxs])
    idxs = np.where(dists==np.sort(dists.reshape(-1))[1])
    zero_img, one_img = X[idxs[0][0]], X[idxs[1][0]]
    
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax1.imshow(zero_img.reshape(28, 28))

    ax2 = fig.add_subplot(2,1,2)
    ax2.imshow(one_img.reshape(28, 28))

    fig.savefig('figure.png')



if __name__ == "__main__":
    app()
