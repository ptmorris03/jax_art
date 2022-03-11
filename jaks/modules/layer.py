from dataclasses import dataclass

from .core import Module
from .nonlin import RELU
from .transform import Rotate, Translate, Scale, ZScore


@dataclass
class Linear(Module):
    in_dims: int
    out_dims: int

    def modules(self):
        yield "weight", Rotate(self.in_dims, self.out_dims)
        yield "bias", Translate(self.out_dims)


@dataclass
class LayerNorm(Module):
    dims: int
    epsilon: float = 1e-05
    
    def modules(self):
        yield "zscore", ZScore(self.epsilon)
        yield "scale", Scale(self.dims)
        yield "translate", Translate(self.dims)


@dataclass
class MLP(Module):
    dims: Iterable[int]
    act_module: Module = RELU()
    
    def modules(self):
        for i in range(len(self.dims) - 1):
            if i > 0:
                yield "activation", self.act_module
            yield F"linear{i + 1}", Linear(self.dims[i], self.dims[i + 1])
