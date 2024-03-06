"""Layers useful in different contexts."""

from typing import Callable
from einops import rearrange
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array, Int, Bool
from flax import linen as nn

from avid.config import MainConfig
from avid.dataset import DataBatch, load_file
from avid.utils import ELEM_VALS, debug_structure, flax_summary, tcheck


class Identity:
    """Module that does nothing, used so model summaries accurately communicate the purpose."""

    @nn.compact
    def __call__(self, x):
        return x


class LazyInOutMLP(nn.Module):
    """Customizable MLP with in and out dimensions given at runtime."""

    inner_dims: tuple[int]
    inner_act: Callable = nn.relu
    final_act: Callable = Identity()
    norm: Callable = nn.LayerNorm
    dropout_rate: float = 0.0
    kernel_init: Callable = nn.initializers.glorot_normal()
    bias_init: Callable = nn.initializers.truncated_normal()

    @tcheck
    @nn.compact
    def __call__(
        self, x: Float[Array, 'n_in'], out_dim: int, training: bool
    ) -> Float[Array, 'out_dim']:
        _curr_dim = x.shape[-1]
        for next_dim in self.inner_dims:
            x = nn.Dense(next_dim, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
            x = self.inner_act(x)
            x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)
            x = self.norm()(x)
            _curr_dim = next_dim

        x = nn.Dense(out_dim, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        x = self.final_act(x)
        return x
