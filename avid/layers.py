"""Layers useful in different contexts."""

from typing import Callable, Optional

from flax import linen as nn
from jaxtyping import Array, Float

from avid.utils import tcheck


class Identity:
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
            x = nn.Dense(next_dim, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
            x = self.inner_act(x)
            x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)
            x = nn.LayerNorm()(x)
            _curr_dim = next_dim

        x = nn.Dense(out_dim, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        x = self.final_act(x)
        return x
