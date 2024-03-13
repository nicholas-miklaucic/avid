"""Checkpointing utils."""

from os import PathLike
from pathlib import Path
import pyrallis
from avid.config import MainConfig
import orbax.checkpoint as ocp
from avid.e_form_predictor import Checkpoint, TrainState, TrainingRun

from avid.utils import debug_structure


def best_ckpt(run_dir: PathLike) -> Checkpoint:
    with open(Path(run_dir) / 'config.toml') as conf_file:
        config = pyrallis.cfgparsing.load(MainConfig, conf_file)

    mngr = ocp.CheckpointManager(
        run_dir.absolute() / 'final_ckpt' / 'ckpts',
        ocp.StandardCheckpointer(),
        options=ocp.CheckpointManagerOptions(
            enable_async_checkpointing=False,
            read_only=True,
            save_interval_steps=0,
            create=False,
        ),
    )

    run = TrainingRun(config)

    model = mngr.restore(mngr.best_step(), args=ocp.args.StandardRestore(run.ckpt()))
    return model


if __name__ == '__main__':
    run_dir = Path('logs') / 'e_form_mae_481'

    with open(run_dir / 'config.toml') as conf_file:
        config = pyrallis.cfgparsing.load(MainConfig, conf_file)

    mngr = ocp.CheckpointManager(
        run_dir.absolute() / 'final_ckpt' / 'ckpts',
        ocp.StandardCheckpointer(),
        options=ocp.CheckpointManagerOptions(
            enable_async_checkpointing=False,
            read_only=True,
            save_interval_steps=0,
            create=False,
        ),
    )

    run = TrainingRun(config)

    model = mngr.restore(mngr.best_step(), args=ocp.args.StandardRestore(run.ckpt()))
    debug_structure(model=model)
