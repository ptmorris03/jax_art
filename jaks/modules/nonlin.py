from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from .core import Module


class RELU(Module):
    def forward(self, params: Any, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.maximum(x, 0)


class QuickGELU(Module):
    def forward(self, params: Any, x: jnp.ndarray) -> jnp.ndarray:
        return x * jax.nn.sigmoid(1.702 * x)


@dataclass
class ZScore(Module):
    epsilon: float = 1e-05

    def forward(self, params: Any, x: jnp.ndarray) -> jnp.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True) + self.epsilon
        return (x - mean) / std
