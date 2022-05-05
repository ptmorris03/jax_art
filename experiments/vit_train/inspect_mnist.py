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
    idxs = np.where(dists==dists.min())
    print(idxs)
    zero_img, one_img = X[zero_idxs[idxs[0][0]]], X[one_idxs[idxs[1][0]]]

    midpoint_img = (zero_img + one_img) / 2
    radius_img = np.minimum(np.abs(midpoint_img - zero_img), np.abs(midpoint_img - one_img))

    batch_n = 100000
    n_batch = 10
    r = 1

    ball_imgs = midpoint_img + np.random.uniform(-r, r, size=(batch_n * n_batch, 784)) * radius_img

    cls_idxs = np.zeros(batch_n * n_batch, dtype=int)
    for batch_idx in range(0, ball_imgs.shape[0], batch_n):
        ball_batch = ball_imgs[batch_idx:batch_idx+batch_n]
        cls_idxs[batch_idx:batch_idx+batch_n] = forward_fn(params, ball_batch.reshape(-1, 1, 28, 28)).argmax(axis=-1)
    ball_zero = cls_idxs == 0
    ball_one = cls_idxs == 1
    ball_other = cls_idxs >= 2
    
    fig = plt.figure()
    ax = fig.add_subplot(2,2,1)
    ax.imshow(zero_img.reshape(28, 28))

    ax = fig.add_subplot(2,2,2)
    ax.imshow(midpoint_img.reshape(28, 28))

    ax = fig.add_subplot(2,2,3)
    ax.imshow(ball_imgs[5].reshape(28, 28))

    ax = fig.add_subplot(2,2,4)
    ax.imshow(one_img.reshape(28, 28))

    fig.savefig('figure.png')

    #pca = PCA(2).fit(ball_imgs[:batch_n])
    pca = PCA(2).fit(np.stack([zero_img, midpoint_img, one_img]))
    ball_proj = pca.transform(ball_imgs)
    zero_proj = pca.transform(zero_img.reshape(1, -1))
    one_proj = pca.transform(one_img.reshape(1, -1))
    midpoint_proj = pca.transform(midpoint_img.reshape(1, -1))
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(ball_proj[ball_one,0], ball_proj[ball_one,1], color='red', s=.01, label="one")
    ax.scatter(ball_proj[ball_zero,0], ball_proj[ball_zero,1], color='blue', s=.1, label="zero")
    ax.scatter(ball_proj[ball_other,0], ball_proj[ball_other,1], color='green', s=10, label="other")
    ax.scatter(zero_proj[:,0], zero_proj[:,1], color='cyan', s=100, label='Actual Zero')
    ax.scatter(one_proj[:,0], one_proj[:,1], color='maroon', s=100, label='Actual One')
    ax.scatter(midpoint_proj[:,0], midpoint_proj[:,1], color='black', s=100, label="ACtual Midpoint")
    plt.legend()
    plt.gcf().set_size_inches(20, 20)
    fig.savefig('scatter.png')



if __name__ == "__main__":
    app()
