import logging
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import optax
import pyrallis
from flax import linen as nn
from flax.struct import dataclass
from pyrallis import field

from avid import layers
from avid.diffusion import DiffusionModel, UNet
from avid.encoder import Downsample, ReduceSpeciesEmbed, SpeciesEmbed
from avid.layers import EquivariantMixerMLP, Identity, LazyInMLP, MLPMixer
from avid.utils import ELEM_VALS
from avid.vit import (
    AddPositionEmbs,
    Encoder,
    ImageEmbed,
    MLPMixerRegressor,
    O3ImageEmbed,
    SingleImageEmbed,
    ViTRegressor,
)

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

    # Seed for dataset shuffling. Controls the order batches are given to train.
    shuffle_seed: int = 1618

    # Train split.
    train_split: int = 8
    # Test split.
    test_split: int = 3
    # Valid split.
    valid_split: int = 3

    # Data augmentations
    # If False, disables all augmentations.
    do_augment: bool = True
    # Random seed for augmentations.
    augment_seed: int = 12345
    # Whether to apply SO(3) augmentations: proper rotations
    so3: bool = True
    # Whether to apply O(3) augmentations: includes SO(3) and also reflections.
    o3: bool = True
    # Whether to apply T(3) augmentations: origin shifts.
    t3: bool = True


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

        pretty_install(crop=True)
        traceback_install()

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

    def __post_init__(self):
        import os

        os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'


@dataclass
class Layer:
    """Serializable layer representation. Works for any named layer in layers.py or flax.nn."""

    # The name of the layer.
    name: str

    def build(self) -> Callable:
        """Makes a new layer with the given values, or returns the function if it's a function."""
        if self.name == 'Identity':
            return Identity()

        for module in (nn, layers):
            if hasattr(module, self.name):
                layer = getattr(module, self.name)
                if isinstance(layer, nn.Module):
                    return getattr(module, self.name)()
                else:
                    # something like relu
                    return layer

        msg = f'Could not find {self.name} in flax.linen or avid.layers'
        raise ValueError(msg)


@dataclass
class MLPConfig:
    """Settings for MLP configuration."""

    # Inner dimensions for the MLP.
    inner_dims: list[int] = field(default_factory=list)

    # Inner activation.
    activation: Layer = field(default=Layer('relu'))

    # Final activation.
    final_activation: Layer = field(default=Layer('Identity'))

    # Output dimension. None means the same size as the input.
    out_dim: Optional[int] = None

    # Dropout.
    dropout: float = 0.1

    # Whether to make the layer equivariant.
    equivariant: bool = False

    def build(self) -> nn.Module:
        """Builds the head from the config."""
        if self.equivariant:
            return EquivariantMixerMLP(
                num_hidden_layers=len(self.inner_dims),
                dropout_rate=self.dropout,
                activation=self.activation.build(),
            )
        else:
            return LazyInMLP(
                inner_dims=self.inner_dims,
                out_dim=self.out_dim,
                inner_act=self.activation.build(),
                final_act=self.final_activation.build(),
                dropout_rate=self.dropout,
            )


@dataclass
class LogConfig:
    log_dir: Path = Path('logs/')

    exp_name: Optional[str] = None

    # How many times to make a log each epoch.
    # 208 = 2^4 * 13 steps per epoch with batch of 1: evenly dividing this is nice.
    logs_per_epoch: int = 8


@dataclass
class ViTInputConfig:
    """Controls how images are processed for ViT."""

    # Patch size: p, where image is broken up into p x p x p cubes.
    patch_size: int = 4
    # Patch inner dimension.
    patch_latent_dim: int = 384
    # Position embedding initialization. 'legendre' uses custom Legendre basis, concatenating with
    # the patch embedding. 'learned' uses standard learned embeddings that are added with the patch
    # embedding.
    pos_embed_type: str = 'learned'

    def build(self):
        if self.pos_embed_type == 'learned':
            pos_embed = AddPositionEmbs(name='pos_embed')
            return ImageEmbed(
                SingleImageEmbed(
                    patch_size=self.patch_size,
                    patch_latent_dim=self.patch_latent_dim,
                    pos_embed=pos_embed,
                ),
                name='image_embed',
            )
        else:
            raise ValueError('Other position embedding types not implemented yet.')


@dataclass
class ViTEncoderConfig:
    """Settings for vision transformer encoder."""

    # Number of encoder layers.
    num_layers: int = 2

    # Number of heads for MHSA.
    num_heads: int = 4

    # Dropout rate applied to inputs for attention.
    enc_dropout_rate: float = 0.2104
    # MLP config after attention layer. The out dim doesn't matter: it's constrained to the input.
    mlp: MLPConfig = field(default_factory=MLPConfig)

    def build(self) -> Encoder:
        return Encoder(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout_rate=self.enc_dropout_rate,
            attention_dropout_rate=self.enc_attention_dropout_rate,
            mlp=self.mlp.build(),
        )


@dataclass
class SpeciesEmbedConfig:
    # Remember that this network will be applied for every voxel of every input point: it's how the
    # data that downsamplers or any downstream encoders can use is generated. These are very
    # flop-intensive parameters.

    # Embedding dimension of the species.
    species_embed_dim: int = 32
    # MLP that embeds species embed + density to a new embedding.
    spec_embed: MLPConfig = field(default_factory=lambda: MLPConfig(out_dim=64))
    # Whether to use the above MLP config or a simple weighted average.
    use_simple_weighting: bool = False

    def build(self) -> ReduceSpeciesEmbed:
        return ReduceSpeciesEmbed(
            SpeciesEmbed(
                len(ELEM_VALS),
                self.species_embed_dim,
                self.spec_embed.build(),
                self.use_simple_weighting,
            ),
            name='species_embed',
        )


@dataclass
class DownsampleConfig:
    # This is the majority of the runtime!

    # Downsampling factor: if 2, then new image is 1/8 the size.
    factor: list[int] = field(default_factory=lambda: [2])

    # Output channels.
    channels_out: list[int] = field(default_factory=lambda: [256])

    # Kernel size: Fairly flop-expensive to increase. Must be at least 2 * factor - 1 to actually
    # use every input data point.
    kernel_size: list[int] = field(default_factory=lambda: [3])

    def __post_init__(self):
        if not (len(self.factor) == len(self.channels_out) == len(self.kernel_size)):
            msg = f'Configuration {self} has mismatched lengths!'
            raise ValueError(msg)

        for kernel, factor in zip(self.kernel_size, self.factor):
            if (kernel - 1) // 2 < (factor - 1):
                msg = f'Configuration {self} would skip data!'
                raise ValueError(msg)

    def build(self) -> nn.Module:
        if len(self.factor) == 0:
            return Identity()

        downsamples = []
        for factor, channels, kernel in zip(self.factor, self.channels_out, self.kernel_size):
            downsamples.append(Downsample(factor, channels, kernel))
        return nn.Sequential(downsamples, name='downsample')


@dataclass
class ViTRegressorConfig:
    """Composite config for ViT regressor, including all parts."""

    vit_input: ViTInputConfig = field(default_factory=ViTInputConfig)
    encoder: ViTEncoderConfig = field(default_factory=ViTEncoderConfig)
    head: MLPConfig = field(default_factory=MLPConfig)
    species_embed: SpeciesEmbedConfig = field(default_factory=SpeciesEmbedConfig)
    downsample: DownsampleConfig = field(default_factory=DownsampleConfig)
    out_dim: int = 1

    def build(self) -> ViTRegressor:
        head = self.head.build()
        head.out_dim = self.out_dim
        return ViTRegressor(
            spec_embed=self.species_embed.build(),
            downsample=self.downsample.build(),
            head=head,
            encoder=self.encoder.build(),
            im_embed=self.vit_input.build(),
        )


@dataclass
class MLPMixerConfig:
    """Composite config for MLP Mixer regressor, including all parts."""

    # Patch size: p, where image is broken up into p x p x p cubes.
    patch_size: int = 3
    # Number of inner transformations to apply for each patch.
    patch_heads: int = 8
    # Patch inner dimension.
    patch_latent_dim: int = 512
    token_mixer: MLPConfig = field(
        default=MLPConfig(activation=Layer('gelu'), equivariant=True), is_mutable=True
    )
    channel_mixer: MLPConfig = field(default_factory=lambda: MLPConfig(activation=Layer('gelu')))
    num_blocks: int = 4
    head: MLPConfig = field(default_factory=MLPConfig)
    species_embed: SpeciesEmbedConfig = field(default_factory=SpeciesEmbedConfig)
    downsample: DownsampleConfig = field(default_factory=DownsampleConfig)
    out_dim: int = 1

    def build(self) -> MLPMixerRegressor:
        head = self.head.build()
        head.out_dim = self.out_dim
        return MLPMixerRegressor(
            spec_embed=self.species_embed.build(),
            downsample=self.downsample.build(),
            head=head,
            mixer=MLPMixer(
                out_dim=self.out_dim,
                num_blocks=self.num_blocks,
                tokens_mlp=self.token_mixer.build(),
                channels_mlp=self.channel_mixer.build(),
            ),
            im_embed=ImageEmbed(
                O3ImageEmbed(self.patch_size, self.patch_latent_dim, self.patch_heads, Identity())
            ),
        )


@dataclass
class UNetConfig:
    """UNet configuration."""

    # Output channels for each block.
    widths: list[int] = field(default_factory=lambda: [8, 16])

    # Number of residual blocks per layer.
    block_depth: int = 1

    # Positional encoding dimension.
    embed_dim: int = 8

    # The minimum sinusoidal frequency.
    embed_min_freq: float = 1.0

    # The maximum frequency.
    embed_max_freq: float = 1000.0

    def build(self) -> UNet:
        return UNet(
            widths=self.widths,
            block_depth=self.block_depth,
            embed_dims=self.embed_dim,
            embed_max_freq=self.embed_max_freq,
            embed_min_freq=self.embed_min_freq,
        )


@dataclass
class DiffusionConfig:
    """Diffusion model configuration."""

    # Minimum signal.
    min_signal_rate: float = 0.02
    # Maximum signal.
    max_signal_rate: float = 0.95


@dataclass
class DiffusionUNetConfig:
    """Config for diffusion using UNet."""

    unet: UNetConfig = field(default_factory=UNetConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)

    def build(self) -> DiffusionModel:
        return DiffusionModel(
            min_signal_rate=self.diffusion.min_signal_rate,
            max_signal_rate=self.diffusion.max_signal_rate,
            model=self.unet.build(),
        )


@dataclass
class LossConfig:
    """Config defining the loss function."""

    # delta for smooth_l1_loss. delta = 0 is L1 loss, and high delta behaves like L2 loss.
    loss_delta: float = 0.1

    # Whether to use RMSE loss instead.
    use_rmse: bool = False

    def regression_loss(self, preds, targets):
        return jax.lax.cond(
            self.use_rmse,
            lambda: jnp.sqrt(optax.losses.squared_error(preds, targets).mean()),
            lambda: (
                optax.losses.huber_loss(preds, targets, delta=self.loss_delta) / self.loss_delta
            ).mean(),
        )


@dataclass
class TrainingConfig:
    """Training/optimizer parameters."""

    # Loss function.
    loss: LossConfig = field(default_factory=LossConfig)

    # Learning rate schedule: 'cosine' for warmup+cosine annealing, 'finder' for a linear schedule
    # that goes up to 20 times the base learning rate.
    lr_schedule_kind: str = 'cosine'

    # Initial learning rate, as a fraction of the base LR.
    start_lr_frac: float = 0.1

    # Base learning rate.
    base_lr: float = 4e-3

    # Final learning rate, as a fraction of the base LR.
    end_lr_frac: float = 0.04

    # Weight decay. AdamW interpretation, so multiplied by the learning rate.
    weight_decay: float = 0.03

    # Beta 1 for Adam.
    beta_1: float = 0.9

    # Beta 2 for Adam.
    beta_2: float = 0.999

    # Nestorov momentum.
    nestorov: bool = True

    # Gradient norm clipping.
    max_grad_norm: float = 1.0

    def lr_schedule(self, num_epochs: int, steps_in_epoch: int):
        if self.lr_schedule_kind == 'cosine':
            warmup_steps = steps_in_epoch * min(5, num_epochs // 2)
            return optax.warmup_cosine_decay_schedule(
                init_value=self.base_lr * self.start_lr_frac,
                peak_value=self.base_lr,
                warmup_steps=warmup_steps,
                decay_steps=num_epochs * steps_in_epoch,
                end_value=self.base_lr * self.end_lr_frac,
            )
        else:
            raise ValueError('Other learning rate schedules not implemented yet')

    def optimizer(self, learning_rate):
        return optax.chain(
            optax.adamw(
                learning_rate,
                b1=self.beta_1,
                b2=self.beta_2,
                weight_decay=self.weight_decay,
                nesterov=self.nestorov,
            ),
            optax.clip_by_global_norm(self.max_grad_norm),
        )


@dataclass
class MainConfig:
    # The batch size. Should be a multiple of data_batch_size to make data loading simple.
    batch_size: int = 52 * 1

    # Use profiling.
    do_profile: bool = False

    # Number of epochs.
    num_epochs: int = 100

    # Folder to initialize all parameters from, if the folder exists.
    restart_from: Optional[Path] = None

    # Folder to initialize the encoders and downsampling.
    encoder_start_from: Optional[Path] = None

    voxelizer: VoxelizerConfig = field(default_factory=VoxelizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    cli: CLIConfig = field(default_factory=CLIConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    log: LogConfig = field(default_factory=LogConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    vit: ViTRegressorConfig = field(default_factory=ViTRegressorConfig)
    diffusion: DiffusionUNetConfig = field(default_factory=DiffusionUNetConfig)
    mlp: MLPMixerConfig = field(default_factory=MLPMixerConfig)

    regressor: str = 'vit'

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

    def build_vit(self) -> ViTRegressor:
        return self.vit.build()

    def build_diffusion(self) -> DiffusionModel:
        return self.diffusion.build()

    def build_mlp(self) -> MLPMixerRegressor:
        return self.mlp.build()

    def build_regressor(self):
        if self.regressor == 'vit':
            return self.build_vit()
        elif self.regressor == 'mlp':
            return self.build_mlp()


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
