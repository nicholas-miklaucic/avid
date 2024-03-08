"""Training run file."""

from pyrallis import wrap

from avid.config import MainConfig
from avid.dataset import load_file
from avid.e_form_predictor import TrainingRun, ViTRegressor
from avid.training_runner import run_using_dashboard
from avid.utils import debug_structure, flax_summary
import jax

@wrap()
def train_e_form(config: MainConfig):
    """Trains the encoder/ViT to predict formation energy."""
    run = TrainingRun(config)
    run_using_dashboard(config, run)


if __name__ == '__main__':
    train_e_form()