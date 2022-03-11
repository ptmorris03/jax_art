from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Union, Tuple, Callable, Any, Iterable

import jax
import jax.numpy as jnp
from jax.random import PRNGKey

from .core import Module


@dataclass
class Rotate(Module):
    in_dims: int
    out_dims: int
    init_scale: float = 1e-2

    def parameters(self, random_key: PRNGKey):
        matrix = jax.random.normal(random_key, (self.out_dims, self.in_dims))
        yield "matrix", matrix * self.init_scale

    def forward(self, params: OrderedDict, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.matmul(params["matrix"], x)
