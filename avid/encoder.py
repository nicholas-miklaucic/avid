"""Encoder/decoder for crystal data."""

from functools import partial
from typing import Callable, Optional
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array, jaxtyped, PRNGKeyArray, Int
from beartype import beartype as typechecker
from flax import linen as nn

from avid.config import MainConfig
from avid.dataset import load_file
from avid.utils import debug_structure, flax_summary, tcheck


class LazyInOutMLP(nn.Module):
    inner_dims: tuple[int]
    inner_act: Callable = nn.relu
    final_act: Callable = lambda x: x
    norm: nn.Module = nn.LayerNorm()
    dropout_rate: float = 0.0
    kernel_init: Callable = nn.initializers.glorot_normal
    bias_init: Callable = nn.initializers.truncated_normal

    @tcheck
    @nn.compact
    def __call__(
        self, x: Float[Array, '*batch n_in'], out_dim: int, training: bool
    ) -> Float[Array, '*batch {out_dim}']:
        _curr_dim = x.shape[-1]
        for next_dim in self.inner_dims:
            x = nn.DenseGeneral(
                next_dim, kernel_init=self.kernel_init, bias_init=self.bias_init, axis=-1
            )(x)
            x = self.inner_act(x)
            x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)
            x = self.norm(x)
            _curr_dim = next_dim

        x = nn.DenseGeneral(
            out_dim, kernel_init=self.kernel_init, bias_init=self.bias_init, axis=-1
        )(x)
        x = self.final_act(x)
        return x


class SpeciesEmbed(nn.Module):
    """Species density embedding."""

    n_species: int
    species_embed_dim: int
    dim_out: int
    embed_module: LazyInOutMLP

    @tcheck
    @nn.compact
    def __call__(
        self, x: Float[Array, ''], spec: Int[Array, ''], training: bool
    ) -> Float[Array, '{self.dim_out}']:
        spec_embed = nn.Embed(self.n_species, self.species_embed_dim, name='species_embed')

        input_embeds = jnp.concat([spec_embed(spec), x], axis=-1)
        return self.embed_module(input_embeds, training=training, out_dim=self.dim_out)


class Downsample(nn.Module):
    downsample_factor: int
    channel_out: int
    kernel_size: int
    kernel_init: Callable = nn.initializers.truncated_normal(stddev=0.1)
    bias_init: Callable = nn.initializers.truncated_normal()

    def setup(self):
        if (self.kernel_size - 1) // 2 < (self.downsample_factor - 1):
            raise ValueError(f'Configuration {self} would skip data!')

        self.conv = nn.Conv(
            features=self.channel_out,
            kernel_size=[self.kernel_size for _ in range(3)],
            strides=[self.downsample_factor for _ in range(3)],
            padding='CIRCULAR',
            name='conv',
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )

    @tcheck
    def __call__(
        self, x: Float[Array, '*batch D H W C']
    ) -> Float[
        Array,
        '*batch D//{self.downsample_factor} H//{self.downsample_factor} W//{self.downsample_factor} {self.channel_out}',
    ]:
        return self.conv(x)


if __name__ == '__main__':
    config = MainConfig()
    embed_mlp = LazyInOutMLP(
        inner_dims=(32,),
    )
    BatchDense = nn.vmap(
        nn.Dense,
        in_axes=0,
        out_axes=0,
        variable_axes={'params': None},
        split_rngs={'params': False},
    )
    spec_embed = ReduceSpeciesEmbed(species_embed_dim=16, dim_out=8, embed_module=embed_mlp)
    downsample = Downsample(downsample_factor=2, channel_out=10, kernel_size=3)
    key = jax.random.key(12345)

    batch = load_file(config, 0)

    kwargs = dict(training=True)

    input = batch.density
    input0 = input.reshape(-1, input.shape[-1])[0]
    params = spec_embed.init(key, input0, **kwargs)
    apply = lambda x: spec_embed.apply(params, x, **kwargs)
    out = nn.BatchApply(jax.vmap(apply), len(input.shape) - 1)(input)

    debug_structure(module=spec_embed)
    flax_summary(spec_embed, input0, **kwargs)
