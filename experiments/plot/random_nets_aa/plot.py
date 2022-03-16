import jaks
import jaks.modules as nn
from tqdm import tqdm
import typer

from dataclasses import dataclass
import json
from pathlib import Path


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


app = typer.Typer()


@app.command()
def run(config: Path):
    cfg = json.load(config.open('r'))


if __name__ == "__main__":
    app()
