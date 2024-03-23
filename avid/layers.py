"""Layers useful in different contexts."""

from typing import Callable, Optional

import einops
import jax.numpy as jnp
from flax import linen as nn
from jaxtyping import Array, Float

from avid.utils import tcheck


class Identity(nn.Module):
    """Module that does nothing, used so model summaries accurately communicate the purpose."""

    @nn.compact
    def __call__(self, x):
        return x


class LazyInMLP(nn.Module):
    """Customizable MLP with input dimension inferred at runtime."""

    inner_dims: tuple[int]
    out_dim: Optional[int]
    inner_act: Callable = nn.relu
    final_act: Callable = Identity()
    dropout_rate: float = 0.0
    kernel_init: Callable = nn.initializers.glorot_normal()
    bias_init: Callable = nn.initializers.truncated_normal()

    @tcheck
    @nn.compact
    def __call__(self, x: Float[Array, 'n_in'], training: bool):
        _curr_dim = x.shape[-1]
        if self.out_dim is None:
            out_dim = _curr_dim
        else:
            out_dim = self.out_dim

        for next_dim in self.inner_dims:
            x = nn.Dense(
                next_dim, kernel_init=self.kernel_init, bias_init=self.bias_init, dtype=jnp.bfloat16
            )(x)
            x = self.inner_act(x)
            x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)
            x = nn.LayerNorm()(x)
            _curr_dim = next_dim

        x = nn.Dense(
            out_dim, kernel_init=self.kernel_init, bias_init=self.bias_init, dtype=jnp.bfloat16
        )(x)
        x = self.final_act(x)
        return x


# 8^6 x subspace_dim
Q = jnp.load('precomputed/equiv_basis_8.npy')
ng6, subspace_dim = Q.shape
ng = round(ng6 ** (1 / 6))
assert ng**6 == ng6
Q = Q.reshape(ng**3, ng**3, subspace_dim).astype(jnp.bfloat16)
conv_kernel = Q[0, ...].reshape(ng, ng, ng, 35)
counts = jnp.sum(conv_kernel, axis=(0, 1, 2))


class EquivariantLinear(nn.Module):
    kernel_init: Callable = nn.initializers.normal(stddev=1e-2, dtype=jnp.bfloat16)

    @nn.compact
    def __call__(self, x, out_head_mul: int):
        batch, out_dim, chan = x.shape
        assert out_dim**2 == ng6, f'{x.shape} is not valid!'

        conv = nn.Conv(
            features=chan,
            kernel_size=(ng, ng, ng),
            use_bias=False,
            padding='CIRCULAR',
            dtype=jnp.bfloat16,
        )

        # x_im = einops.rearrange(
        #     x, 'batch (n1 n2 n3) chan -> (batch chan) n1 n2 n3 1', n1=ng, n2=ng, n3=ng
        # )
        x_im = einops.rearrange(
            x, 'batch (n1 n2 n3) chan -> batch n1 n2 n3 chan', n1=ng, n2=ng, n3=ng
        )
        kernel = self.param('kernel', self.kernel_init, (subspace_dim, chan, chan))
        # QK is out_dim x out_dim
        # is this the most efficient way to do this?
        kernel_expanded = einops.einsum(
            conv_kernel, kernel, 'a b c s, s ch chout -> a b c ch chout'
        )
        conv_out = conv.apply({'params': {'kernel': kernel_expanded}}, x_im)
        out = einops.rearrange(conv_out, 'batch n1 n2 n3 chan -> batch (n1 n2 n3) chan')
        return out


class EquivariantMixerMLP(nn.Module):
    activation: Callable = nn.gelu
    dropout_rate: float = 0.0
    head_muls: tuple[int] = (2, 1)

    @nn.compact
    def __call__(self, x, training: bool = False):
        for head_mul in self.head_muls:
            linear = EquivariantLinear()
            x = linear(x, head_mul)
            x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)
            x = nn.LayerNorm(dtype=x.dtype, use_bias=False, use_scale=False)(x)
            x = self.activation(x)

        return x


class MixerBlock(nn.Module):
    """
    A Flax Module to act as the mixer block layer for the MLP-Mixer Architecture.

    Attributes:
        tokens_mlp_dim: MLP Block 1
        channels_mlp_dim: MLP Block 2
    """

    tokens_mlp: nn.Module
    channels_mlp: LazyInMLP

    @nn.compact
    def __call__(self, x, training: bool = False) -> Array:
        # Layer Normalization
        y = nn.LayerNorm(dtype=x.dtype, use_bias=False, use_scale=False)(x)
        # # Transpose
        # y = jnp.swapaxes(y, 1, 2)
        # MLP 1
        y = self.tokens_mlp(y, training)
        # Transpose
        # y = jnp.swapaxes(y, 1, 2)
        # Skip Connection
        # now it's possible that the channels have changed
        # if so, broadcast x to fit
        x = x[..., None] + y.reshape(*x.shape, -1)
        x = einops.rearrange(x, 'batch dim chan chan_mul -> batch dim (chan chan_mul)')
        # Layer Normalization
        y = nn.LayerNorm(dtype=x.dtype, use_bias=False, use_scale=False)(x)
        # MLP 2 with Skip Connection
        out = x + self.channels_mlp(y, training)
        return out


class MLPMixer(nn.Module):
    """
    Flax Module for the MLP-Mixer Architecture.

    Attributes:
        patches: Patch configuration
        num_classes: No of classes for the output head
        num_blocks: No of Blocks of Mixers to use
        hidden_dim: No of Hidden Dimension for the Patch-Wise Convolution Layer
        tokens_mlp_dim: No of dimensions for the MLP Block 1
        channels_mlp_dim: No of dimensions for the MLP Block 2
        approximate: If True, uses the approximate formulation of GELU in each MLP Block
        dtype: the dtype of the computation (default: float32)
    """

    out_dim: int
    num_blocks: int
    tokens_mlp: nn.Module
    channels_mlp: LazyInMLP

    @nn.compact
    def __call__(self, inputs, *, training: bool = False) -> Array:
        x = inputs
        # Num Blocks x Mixer Blocks
        for _ in range(self.num_blocks):
            x = MixerBlock(
                tokens_mlp=self.tokens_mlp.copy(),
                channels_mlp=self.channels_mlp.copy(),
            )(x, training=training)
        # Output Head
        x = nn.LayerNorm(
            dtype=x.dtype, name='pre_head_layer_norm', use_bias=False, use_scale=False
        )(x)
        return x


class PermInvariantEncoder(nn.Module):
    """A la Deep Sets, constructs a permutation-invariant representation of the inputs based on aggregations.
    Uses mean, std, and differentiable quantiles."""

    @nn.compact
    def __call__(self, x):
        """x: batch chan token
        Invariant over the order of tokens."""

        x_mean = jnp.mean(x, axis=-1, keepdims=True)
        x_std = jnp.std(x, axis=-1, keepdims=True)

        # x_whiten = (x - x_mean) / (x_std + 1e-8)

        # x_quants = []
        # for power in jnp.linspace(1, 3, 6):
        #     x_quants.append(
        #         jnp.mean(
        #             jnp.sign(x_whiten) * jnp.abs(x_whiten**power),
        #             axis=-1,
        #             keepdims=True,
        #         )
        #         ** (1 / power),
        #     )

        # This has serious numerical stability problems in the backward pass. Instead, I'll use something else.
        # eps = 0.02
        # quants = jnp.linspace(eps, 1 - eps, 14, dtype=jnp.bfloat16)
        # from ott.tools.soft_sort import quantile
        # x_quants = quantile(x, quants, axis=-1, weight=10 / x.shape[-1])
        return jnp.concat([x_mean, x_std], axis=2)
