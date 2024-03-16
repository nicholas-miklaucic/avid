import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from avid.utils import tcheck


@tcheck
class DataBatch(eqx.Module):
    """A batch of data."""

    density: Float[Array, 'batch n_grid n_grid n_grid max_spec']
    species: Int[Array, 'batch max_spec']
    mask: Bool[Array, 'batch max_spec']
    e_form: Float[Array, 'batch']
    lat_abc_angles: Float[Array, 'batch 6']

    @classmethod
    def new_empty(cls, batch_size: int, n_grid: int, max_spec: int):
        return DataBatch(
            jnp.empty((batch_size, n_grid, n_grid, n_grid, max_spec)),
            jnp.empty((batch_size, max_spec), dtype=jnp.int16),
            jnp.empty((batch_size, max_spec), dtype=jnp.bool),
            jnp.empty(batch_size),
            jnp.empty((batch_size, 6)),
        )

    def device_put(self, devices: jax.sharding.PositionalSharding | jax.Device):
        if isinstance(devices, jax.Device):
            return jax.device_put(self, devices)
        else:
            for key in ['density', 'species', 'mask', 'e_form', 'lat_abc_angles']:
                sh = getattr(self, key).shape
                new_shape = [1] * len(sh)
                new_shape[0] = -1
                jax.device_put(getattr(self, key), devices.reshape(*new_shape))
            return self
