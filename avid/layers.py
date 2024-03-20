"""Layers useful in different contexts."""

from typing import Callable, Optional

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


class MixerBlock(nn.Module):
    """
    A Flax Module to act as the mixer block layer for the MLP-Mixer Architecture.

    Attributes:
        tokens_mlp_dim: MLP Block 1
        channels_mlp_dim: MLP Block 2
    """

    tokens_mlp: LazyInMLP
    channels_mlp: LazyInMLP

    @nn.compact
    def __call__(self, x, training: bool = False) -> Array:
        # Layer Normalization
        y = nn.LayerNorm(dtype=x.dtype)(x)
        # Transpose
        y = jnp.swapaxes(y, 1, 2)
        # MLP 1
        y = self.tokens_mlp(y, training)
        # Transpose
        y = jnp.swapaxes(y, 1, 2)
        # Skip Connection
        x = x + y
        # Layer Normalization
        y = nn.LayerNorm(dtype=x.dtype)(x)
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
    tokens_mlp: LazyInMLP
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
        x = nn.LayerNorm(dtype=x.dtype, name='pre_head_layer_norm')(x)
        return x
