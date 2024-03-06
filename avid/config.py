from enum import Enum
import logging
from flax.struct import dataclass
from typing import Literal
from jax import Device
import jax
import pyrallis
from pyrallis import field
from pathlib import Path

pyrallis.set_config_type('toml')


@dataclass
class VoxelizerConfig:
    """Controls how crystals are encoded into a grid."""

    # The max number of species in a single sample. Due to batching, this affects performance and
    # memory consumption.
    max_unique_species: int = 5

    # The grid size per axis.
    n_grid: int = 24

    # The power to raise the distance to when computing probabilities. 2 is a Gaussian distribution,
    # 1 is Laplace, and higher values produce tighter spheres around the center.
    distance_power: float = 2.0

    # Constant multiplied by the atom radius to get the standard deviation of the generalized
    # normal.
    eff_scale: float = 0.7

    # The number of cells in each direction to extend outwards for computing the probabilities.
    # Higher values are more numerically precise but take longer to process, although there's no
    # impact on the runtime of training or inference beyond the preprocessing step.
    num_cells: int = 4

    @property
    def n_points(self) -> int:
        """The total number of points in a single unit cell."""
        return self.n_grid**3


@dataclass
class DataConfig:
    # The batch size for processing. The dataset size is 2^4 x 7 x 13^2, so 52 is a reasonable
    # value that I've picked to make even batches.
    data_batch_size: int = 52

    # Folder of raw data files.
    raw_data_folder: Path = Path('data/')

    # Folder of processed data files.
    data_folder: Path = Path('precomputed/')

    # Train split.
    train_split: int = 6
    # Test split.
    test_split: int = 0
    # Valid split:
    valid_split: int = 1


class LoggingLevel(Enum):
    """The logging level."""

    debug = logging.DEBUG
    info = logging.INFO
    warning = logging.WARNING
    error = logging.ERROR
    critical = logging.CRITICAL


@dataclass
class CLIConfig:
    # Verbosity of output.
    verbosity: LoggingLevel = LoggingLevel.info
    # Whether to show progress bars.
    show_progress: bool = True

    def set_up_logging(self):
        from rich.logging import RichHandler
        from rich.pretty import install as pretty_install
        from rich.traceback import install as traceback_install

        pretty_install()
        traceback_install()

        logging.basicConfig(
            level=self.verbosity.value,
            format='%(message)s',
            datefmt='[%X]',
            handlers=[
                RichHandler(
                    rich_tracebacks=True,
                    show_time=False,
                    show_level=False,
                    show_path=False,
                )
            ],
        )


@dataclass
class DeviceConfig:
    device: str = 'gpu'

    @property
    def jax_devices(self):
        return jax.devices(self.device)


@dataclass
class MainConfig:
    # The batch size. Should be a multiple of data_batch_size to make data loading simple.
    batch_size: int = 52 * 1

    # Use profiling.
    do_profile: bool = False

    voxelizer: VoxelizerConfig = field(default_factory=VoxelizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    cli: CLIConfig = field(default_factory=CLIConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)

    def __post_init__(self):
        if self.batch_size % self.data.data_batch_size != 0:
            raise ValueError(
                'Training batch size should be multiple of data batch size: {} does not divide {}'.format(
                    self.batch_size, self.data.data_batch_size
                )
            )

        self.cli.set_up_logging()

    @property
    def train_batch_multiple(self) -> int:
        """How many files should be loaded per training step."""
        return self.batch_size // self.data.data_batch_size


if __name__ == '__main__':
    from pathlib import Path

    from rich.prompt import Confirm

    if Confirm.ask('Generate configs/defaults.toml and configs/minimal.toml?'):
        default_path = Path('configs') / 'defaults.toml'
        minimal_path = Path('configs') / 'minimal.toml'

        default = MainConfig()

        with open(default_path, 'w') as outfile:
            pyrallis.cfgparsing.dump(default, outfile)

        with open(minimal_path, 'w') as outfile:
            pyrallis.cfgparsing.dump(default, outfile, omit_defaults=True)

        with default_path.open('r') as conf:
            pyrallis.cfgparsing.load(MainConfig, conf)
