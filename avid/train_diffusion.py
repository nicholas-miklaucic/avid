"""Training run file."""

from pathlib import Path

from pyrallis import wrap

from avid.config import MainConfig
from avid.training_state import DiffusionDataParallelTrainer


@wrap()
def train_diffusion(config: MainConfig):
    """Trains the diffusion U-Net."""
    model = DiffusionDataParallelTrainer(config, Path('logs') / 'mlp' / 'weights')
    model.train()


if __name__ == '__main__':
    # with jax.log_compiles():
    train_diffusion()
