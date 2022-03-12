import jax
import jax.numpy as jnp

from collections import OrderedDict
import json


def pretty_params(params: OrderedDict, indent: int = 4) -> str:
    shape_map = jax.tree_map(lambda x: str(x.shape), params)
    return json.dumps(shape_map, indent=indent, sort_keys=False)

def log_loss(predictions: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    lse = jax.scipy.special.logsumexp(predictions, axis=-1, keepdims=True)
    log_predictions = predictions - lse
    target_distribution = jax.nn.one_hot(labels, predictions.shape[-1])
    return -jnp.mean(log_predictions * target_distribution)

def accuracy(predictions: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    return predictions.argmax(axis=-1) == labels

def sgd(params: OrderedDict, grads: OrderedDict, lr: float) -> OrderedDict:
    def map_fn(param, grad):
        return param - learning_rate * grad
    return jax.tree_multimap(map_fn, params, grads)
