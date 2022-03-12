from collections import OrderedDict
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .core import Module


@dataclass
class Residual(Module):
    module: Module

    def modules(self):
        yield "residual_module", self.module

    def forward(self, params: OrderedDict, x: jnp.ndarray) -> jnp.ndarray:
        return self.residual_module(params, x) + x


@dataclass
class ResNetStack(Module):
    module: Module
    depth: int

    def modules(self):
        residual_layer = Residual(self.module)
        for layer in range(1, self.depth + 1):
            yield F"layer{layer}", residual


@dataclass
class Loop(Module):
    module: Module
    iterations: int

    def modules(self):
        yield "loop_module", self.module

    def forward(self, params: OrderedDict, x: jnp.ndarray) -> jnp.ndarray:
        def step_fn(i, x):
            return self.loop_module(params, x)
        return jax.lax.fori_loop(0, self.iterations, step_fn, x)
