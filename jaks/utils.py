import jax

from collections import OrderedDict
import json


def pretty_params(params: OrderedDict, indent: int = 4) -> str:
    return json.dumps(
        jax.tree_map(lambda x: str(x.shape), params),
        indent = indent,
        sort_keys = False,
    )
