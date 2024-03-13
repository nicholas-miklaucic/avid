from enum import Enum
import logging

from typing import Optional
from flax.struct import dataclass
from typing import Literal
from jax import Device
import jax
import pyrallis
from pyrallis import field
import orbax
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
        traceback_install(suppress=(orbax))

        logging.basicConfig(
            level=self.verbosity.value,
            format='%(message)s',
            datefmt='[%X]',
            handlers=[
                RichHandler(
                    rich_tracebacks=False,
                    show_time=False,
                    show_level=False,
                    show_path=False,
                )
            ],
        )


@dataclass
class DeviceConfig:
    # Either 'cpu', 'gpu', or 'tpu'
    device: str = 'gpu'

    # Limits the number of GPUs used. 0 means no limit.
    max_gpus: int = 1

    @property
    def jax_device(self):
        devs = jax.devices(self.device)
        if self.device == 'gpu' and self.max_gpus != 0:
            devs = devs[: self.max_gpus]

        if len(devs) > 1:
            return jax.sharding.PositionalSharding(devs)
        else:
            return devs[0]


@dataclass
class LogConfig:
    log_dir: Path = Path('logs/')

    exp_name: Optional[str] = None

    # How many times to make a log each epoch.
    # 208 = 2^4 * 13 steps per epoch with batch of 1: evenly dividing this is nice.
    logs_per_epoch: int = 8


@dataclass
class ViTConfig:
    """Settings for vision transformer."""

    # Patch size: p, where image is broken up into p x p x p cubes.
    patch_size: int = 4
    # Patch inner dimension.
    patch_latent_dim: int = 384
    # Position embedding dimension. Learned positional embeddings don't use this.
    pos_embed_dim: int = 32
    # Position embedding initialization. 'legendre' uses custom Legendre basis, concatenating with
    # the patch embedding. 'learned' uses standard learned embeddings that are added with the patch
    # embedding.
    pos_embed_type: str = 'learned'


@dataclass
class SpeciesEmbedConfig:
    # Inner dimensions of the MLP. This is quite flop-expensive, because it's applied to every voxel
    # of the data before downsampling.
    inner_dims: tuple[int] = ()

    # Output dimension.
    dim_out: int = 32


@dataclass
class DownsampleConfig:
    # Downsampling factor: if 2, then new image is 1/8 the size.
    factor: int = 2

    # Output channels.
    channels_out: int = 256

    # Kernel size: Fairly flop-expensive to increase. Must be at least 2 * factor - 1 to actually
    # use every input data point.
    kernel_size: int = 3

    def __post_init__(self):
        if (self.kernel_size - 1) // 2 < (self.downsample_factor - 1):
            raise ValueError(f'Configuration {self} would skip data!')


@dataclass
class TrainingConfig:
    """Training/optimizer parameters."""

    # Learning rate schedule: 'cosine' for warmup+cosine annealing, 'finder' for a linear schedule
    # that goes up to 20 times the base learning rate.
    lr_schedule: str = 'cosine'

    # Base learning rate.
    base_lr: float = 4e-3


@dataclass
class MainConfig:
    # The batch size. Should be a multiple of data_batch_size to make data loading simple.
    batch_size: int = 52 * 1

    # Use profiling.
    do_profile: bool = False

    # Number of epochs.
    num_epochs: int = 100

    voxelizer: VoxelizerConfig = field(default_factory=VoxelizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    cli: CLIConfig = field(default_factory=CLIConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    log: LogConfig = field(default_factory=LogConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        if self.batch_size % self.data.data_batch_size != 0:
            raise ValueError(
                'Training batch size should be multiple of data batch size: {} does not divide {}'.format(
                    self.batch_size, self.data.data_batch_size
                )
            )

        self.cli.set_up_logging()
        if not self.log.log_dir.exists():
            raise ValueError(f'Log directory {self.log.log_dir} does not exist!')

        from jax.experimental.compilation_cache.compilation_cache import set_cache_dir

        set_cache_dir('/tmp/jax_comp_cache')

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
