"""Simple module to predict e_form from latent representation. A quick way to benchmark an
encoder."""

from typing import Callable
from einops import rearrange, reduce
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array, Int, Bool
from flax import linen as nn
import pyrallis
from rich.prompt import Confirm
from clu import metrics
from flax.training import train_state, orbax_utils
from flax import struct
import optax
import orbax
import orbax.checkpoint as ocp

from avid.config import MainConfig
from avid.coord_embeddings import legendre_embed, legendre_grid_embeds
from avid.dataset import DataBatch, dataloader, load_file
from avid.encoder import Downsample, ReduceSpeciesEmbed, SpeciesEmbed
from avid.layers import LazyInOutMLP, Identity
from avid.utils import ELEM_VALS, debug_structure, flax_summary, tcheck


class Patchify(nn.Module):
    """Patchify block in 3D, with lazy patch size."""

    dim_out: int
    kernel_init: Callable = nn.initializers.glorot_normal()
    bias_init: Callable = nn.initializers.uniform()

    @tcheck
    @nn.compact
    def __call__(self, patch: Float[Array, 'p p p chan_in']) -> Float[Array, '_dimout']:
        patch = rearrange(patch, 'p1 p2 p3 chan_in -> (p1 p2 p3 chan_in)')

        patch_embed = nn.Dense(
            self.dim_out, kernel_init=self.kernel_init, bias_init=self.bias_init, name='patch_proj'
        )

        return patch_embed(patch)


class LegendrePosEmbed(nn.Module):
    """Module to apply fixed Legendre positional encodings."""

    max_input_size: int
    dim_embed: int

    @tcheck
    @nn.compact
    def __call__(
        self, patches: Float[Array, 'I I I patch_d']
    ) -> Float[Array, 'I I I _patchd_posd']:
        embed = self.param(
            'embed', lambda *args: legendre_grid_embeds(patches.shape[0], self.dim_embed)
        )
        step = embed.shape[0] // patches.shape[0]
        return jnp.concat([patches, embed[::step, ::step, ::step, :]], axis=-1)


class SingleImageEmbed(nn.Module):
    """Embeds a 3D image into a sequnce of tokens."""

    patch_size: int
    patch_latent_dim: int
    pos_latent_dim: int
    pos_embed: nn.Module

    @tcheck
    @nn.compact
    def __call__(self, im: Float[Array, 'I I I C']) -> Float[Array, '_ip3 _dimout']:
        i = im.shape[0]
        p = self.patch_size
        c = im.shape[-1]
        assert i % p == 0
        n = i // p

        patches = im.reshape(n, p, n, p, n, p, c)
        patches = rearrange(patches, 'n1 p1 n2 p2 n3 p3 c -> (n1 n2 n3) p1 p2 p3 c')

        embed = jax.vmap(Patchify(self.patch_latent_dim))(patches)
        pos_embed = self.pos_embed(rearrange(embed, '(i1 i2 i3) c -> i1 i2 i3 c', i1=n, i2=n, i3=n))

        return rearrange(pos_embed, 'i1 i2 i3 c -> (i1 i2 i3) c', i1=n, i2=n, i3=n)


class ImageEmbed(nn.Module):
    inner: SingleImageEmbed

    @tcheck
    @nn.compact
    def __call__(self, im: Float[Array, 'batch I I I C']) -> Float[Array, 'batch _ip3 _dim_out']:
        return jax.vmap(self.inner)(im)


class Encoder1DBlock(nn.Module):
    """Transformer encoder layer.

    Attributes:
      inputs: input data.
      mlp_dim: dimension of the mlp on top of attention block.
      dtype: the dtype of the computation (default: float32).
      dropout_rate: dropout rate.
      attention_dropout_rate: dropout for attention heads.
      deterministic: bool, deterministic or not (to apply dropout).
      num_heads: Number of heads in nn.MultiHeadDotProductAttention
    """

    num_heads: int
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, inputs, *, training: bool):
        """Applies Encoder1DBlock module.

        Args:
          inputs: Inputs to the layer.
          deterministic: Dropout will not be applied when set to true.

        Returns:
          output after transformer encoder block.
        """

        # Attention block.
        assert inputs.ndim == 3, f'Expected (batch, seq, hidden) got {inputs.shape}'
        x = nn.LayerNorm()(inputs)
        x = nn.MultiHeadDotProductAttention(
            kernel_init=nn.initializers.xavier_uniform(),
            broadcast_dropout=False,
            deterministic=not training,
            dropout_rate=self.attention_dropout_rate,
            num_heads=self.num_heads,
        )(x, x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
        x = x + inputs

        # MLP block.
        y = nn.LayerNorm()(x)

        mlp = LazyInOutMLP(inner_dims=(), dropout_rate=self.dropout_rate)
        y = jax.vmap(jax.vmap(lambda yy: mlp(yy, out_dim=yy.shape[-1], training=training)))(y)

        return x + y


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation.

    Attributes:
      num_layers: number of layers
      mlp_dim: dimension of the mlp on top of attention block
      num_heads: Number of heads in nn.MultiHeadDotProductAttention
      dropout_rate: dropout rate.
      attention_dropout_rate: dropout rate in self attention.
    """

    num_layers: int
    num_heads: int
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, *, training: bool):
        """Applies Transformer model on the inputs.

        Args:
          x: Inputs to the layer.
          train: Set to `True` when training.

        Returns:
          output of a transformer encoder.
        """
        assert x.ndim == 3  # (batch, len, emb)

        # Input Encoder
        for lyr in range(self.num_layers):
            x = Encoder1DBlock(
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                name=f'encoderblock_{lyr}',
                num_heads=self.num_heads,
            )(x, training=training)
        encoded = nn.LayerNorm(name='encoder_norm')(x)

        return encoded


class ViTRegressor(nn.Module):
    patch_size: int = 4
    patch_latent_dim: int = 32
    pos_latent_dim: int = 16
    max_grid: int = 24
    species_embed_head_inner_dims: tuple[int] = (24,)
    species_embed_dim: int = 16
    species_embed_dim_out: int = 8
    downsample_factors: tuple[int] = (2,)
    channels_out: tuple[int] = (10,)
    kernel_sizes: tuple[int] = (3,)
    num_layers: int = 2
    num_heads: int = 4
    enc_dropout_rate: float = 0.1
    enc_attention_dropout_rate: float = 0.1
    regressor_head_inner_dims: tuple[int] = (48,)
    regressor_head_act: nn.Module = Identity()
    output_dim: int = 1

    @nn.compact
    def __call__(self, im: DataBatch, training: bool):
        pos_embed = LegendrePosEmbed(self.max_grid, self.pos_latent_dim, name='pos_embed')
        im_embed = ImageEmbed(
            SingleImageEmbed(
                self.patch_size, self.patch_latent_dim, self.pos_latent_dim, pos_embed
            ),
            name='image_embed',
        )
        embed_mlp = LazyInOutMLP(inner_dims=self.species_embed_head_inner_dims)
        spec_embed = ReduceSpeciesEmbed(
            SpeciesEmbed(
                len(ELEM_VALS), self.species_embed_dim, self.species_embed_dim_out, embed_mlp
            ),
            name='species_embed',
        )

        downsamples = []
        for downsample_factor, channel_out, kernel_size in zip(
            self.downsample_factors, self.channels_out, self.kernel_sizes
        ):
            downsamples.append(
                Downsample(
                    downsample_factor=downsample_factor,
                    channel_out=channel_out,
                    kernel_size=kernel_size,
                )
            )

        downsample = nn.Sequential(downsamples, name='downsample')

        encoder = Encoder(
            self.num_layers,
            self.num_heads,
            self.enc_dropout_rate,
            self.enc_attention_dropout_rate,
            name='encoder',
        )

        head = LazyInOutMLP(inner_dims=self.regressor_head_inner_dims, name='head')

        out = spec_embed(im, training=training)
        out = downsample(out)
        out = im_embed(out)
        out = encoder(out, training=training)

        out = reduce(out, 'batch seq dim -> batch dim', 'mean')
        # debug_structure(out=out)
        out = jax.vmap(lambda x: head(x, out_dim=self.output_dim, training=training))(out)
        return out


@struct.dataclass
class Metrics(metrics.Collection):
    loss: metrics.Average.from_output('loss')


class TrainState(train_state.TrainState):
    key: jax.Array
    metrics: Metrics


def create_train_state(module, rng, learning_rate):
    params = module.init(rng, DataBatch.new_empty(52, 24, 5), training=False)
    tx = optax.adamw(learning_rate)
    return TrainState.create(
        apply_fn=module.apply, params=params, tx=tx, metrics=Metrics.empty(), key=rng
    )


@jax.jit
def train_step(state: nn.Module, batch: DataBatch, rng):
    """Train for a single step."""
    dropout_train_key = jax.random.fold_in(key=rng, data=state.step)

    def loss_fn(params):
        preds = state.apply_fn(
            params, batch, training=True, rngs={'dropout': dropout_train_key}
        ).squeeze()
        loss = optax.losses.l2_loss(preds, targets=batch.e_form).mean()
        return loss

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state


@jax.jit
def compute_metrics(*, state: TrainState, batch: DataBatch):
    preds = state.apply_fn(state.params, batch, training=False).squeeze()

    loss = optax.losses.l2_loss(preds, targets=batch.e_form).mean()
    metric_updates = state.metrics.single_from_model_output(loss=loss)
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    return state


class TrainingRun:
    def __init__(self, config: MainConfig):
        self.config = MainConfig
        self.orbax_checkpointer = ocp.PyTreeCheckpointer()
        options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)
        self.mngr = orbax.checkpoint.CheckpointManager(
            ocp.test_utils.create_empty('/tmp/ckpt/'), options=options
        )

        self.rng = jax.random.key(27182)
        self.state = create_train_state(ViTRegressor(), self.rng, 1e-3)

        self.metrics_history = {
            'train_loss': [],
            'test_loss': [],
        }

        self.num_epochs = 10
        self.steps_in_epoch, self.dl = dataloader(config, split='train', infinite=True)
        self.steps_in_test_epoch, self.test_dl = dataloader(config, split='valid', infinite=True)
        self.num_steps = self.steps_in_epoch * self.num_epochs
        self.steps = range(self.num_steps)
        self.curr_step = 0

    def next_step(self):
        return self.step(self.curr_step + 1, next(self.dl))

    def step(self, step, batch):
        self.curr_step = step
        if step >= self.num_steps:
            return None

        self.state = train_step(
            self.state, batch, self.rng
        )  # get updated train state (which contains the updated parameters)
        self.state = compute_metrics(state=self.state, batch=batch)  # aggregate batch metrics

        if (step + 1) % self.steps_in_epoch == 0:  # one training epoch has passed
            for metric, value in self.state.metrics.compute().items():  # compute metrics
                self.metrics_history[f'train_{metric}'].append(value.item())  # record metrics
            self.state = self.state.replace(
                metrics=self.state.metrics.empty()
            )  # reset train_metrics for next training epoch

            # Compute metrics on the test set after each training epoch
            self.test_state = self.state
            for _i, test_batch in zip(range(self.steps_in_test_epoch), self.test_dl):
                self.test_state = compute_metrics(state=self.test_state, batch=test_batch)

            for metric, value in self.test_state.metrics.compute().items():
                self.metrics_history[f'test_{metric}'].append(value.item())

        return self

    def step_until_done(self):
        for step, batch in zip(self.steps, self.dl):
            yield self.step(step, batch)

    def run_to_completion(self):
        for step, batch in zip(self.steps, self.dl):
            self.step(step, batch)

    @property
    def epoch_just_finished(self):
        return (self.curr_step + 1) % self.steps_in_epoch == 0


if __name__ == '__main__':
    config = pyrallis.argparsing.parse(MainConfig, 'configs/smoke_test.toml')
    from rich.pretty import pprint
    import rich.progress as prog

    # with jax.profiler.trace('/tmp/jax-trace', create_perfetto_link=True):
    # if config.do_profile:
    #     jax.profiler.start_trace('/tmp/jax-trace', create_perfetto_link=True)

    # run = TrainingRun(config)
    # update_every = 1
    # with prog.Progress(
    #     prog.TextColumn('[progress.description]{task.description}'),
    #     prog.BarColumn(80, 'light_pink3', 'deep_sky_blue4', 'green'),
    #     prog.MofNCompleteColumn(),
    #     prog.TimeElapsedColumn(),
    #     prog.TimeRemainingColumn(),
    #     prog.SpinnerColumn(),
    #     refresh_per_second=3,
    #     disable=not config.cli.show_progress,
    # ) as progress:
    #     task = progress.add_task(
    #         '[bold] [deep_pink3] Training [/deep_pink3] [/bold]',
    #         total=run.num_steps // update_every,
    #     )
    #     for run_state in run.step_until_done():
    #         if run_state.curr_step % update_every == 0:
    #             progress.advance(task)

    #         if run_state.epoch_just_finished:
    #             pprint(run_state.metrics_history)

    # if config.do_profile:
    #     jax.profiler.stop_trace()

    # if not Confirm.ask('Train?'):
    #     raise ValueError('Aborted')

    kwargs = dict(training=False)
    batch = load_file(config, 0)
    mod = ViTRegressor()
    out, params = mod.init_with_output(jax.random.key(0), im=batch, **kwargs)
    debug_structure(batch=batch, module=mod, out=out)
    flax_summary(mod, batch, **kwargs)
