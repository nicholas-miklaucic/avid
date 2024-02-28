"""Module to store different approaches to embed the position."""

import logging
import equinox as eqx
import jax.numpy as jnp
import jax
from jaxtyping import PyTree, Float, Array
import pyrallis
from scipy.special import lpmv
import numpy as np

from avid.config import MainConfig
from avid.utils import debug_stat, debug_structure


def legendre_first_kind(xyz: Float[Array, '3'], m: int, n: int) -> Float[Array, '3']:
    return lpmv(m, n, np.sin(xyz * 2 * np.pi))


def legendre_embed(xyz: Float[Array, 'N 3'], max_n: int) -> Float[Array, 'N 3*{max_n}*{max_n}']:
    """Embeds the positions using Legendre polynomials, using sin for periodicity."""
    dims = []
    for m in range(max_n):
        for n in range(max_n):
            if m <= n:
                dims.append(legendre_first_kind(xyz, m, n))

    dims = jnp.hstack(dims)
    dims = dims / jnp.abs(dims).max(axis=0, keepdims=True)
    return dims


if __name__ == '__main__':
    config = pyrallis.parse(config_class=MainConfig)
    config.cli.set_up_logging()
    from einops import rearrange

    N_GRID = 24
    grid_vals = jnp.linspace(0, 1, N_GRID + 1)[:-1]
    xx, yy, zz = jnp.meshgrid(grid_vals, grid_vals, grid_vals)
    xyz = rearrange(jnp.array([xx, yy, zz]), 'd n1 n2 n3 -> (n1 n2 n3) d')
    embed = legendre_embed(xyz, 4)
    debug_stat(emb=embed)
    debug_structure(emb=embed)
