"""Script to preprocess input data into voxel representation."""

import pyrallis
import jax
import jax.numpy as jnp
import equinox as eqx
import functools as ft
from jaxtyping import Array, Float, Int
from einops import rearrange
import pandas as pd
from rich.prompt import Confirm
from rich.progress import track

from tensorflow_probability.substrates import jax as tfp
from pymatgen.core import Structure

from avid.config import MainConfig
from avid.utils import ELEM_VALS

tfd = tfp.distributions

if __name__ == '__main__':
    if not Confirm.ask('Regenerate data files? This can take a while.'):
        raise ValueError('Aborted')

    config = pyrallis.argparsing.parse(MainConfig, 'configs/defaults.toml')

    # matbench perov dataset
    df = pd.read_json(config.data.raw_data_folder / 'castelli.json', orient='split')
    df['struct'] = [Structure.from_dict(s) for s in df['structure']]

    N_GRID = config.voxelizer.n_grid
    power = config.voxelizer.distance_power
    eff_scale = jnp.array(config.voxelizer.eff_scale)
    num_cells = config.voxelizer.num_cells

    distr = tfd.GeneralizedNormal(0, 1, power)
    grid_vals = jnp.linspace(0, 1, N_GRID + 1)[:-1]
    xx, yy, zz = jnp.meshgrid(grid_vals, grid_vals, grid_vals)
    xyz = rearrange(jnp.array([xx, yy, zz]), 'd n1 n2 n3 -> (n1 n2 n3) d')

    xx_shift, yy_shift, zz_shift = jnp.meshgrid(
        *[jnp.arange(-num_cells, num_cells + 1, dtype=jnp.float32) for _ in range(3)]
    )
    shifts = rearrange(jnp.array([xx_shift, yy_shift, zz_shift]), 'd n1 n2 n3 -> (n1 n2 n3) d')

    @ft.partial(jax.vmap, in_axes=(0, None, None))
    @eqx.filter_jit(donate='none')
    def atom_density(
        xyz: Float[Array, '3'],
        atom_xyz: Float[Array, '3'],
        atom_rad: Float[Array, ''],
    ) -> Float[Array, '']:
        """Atom density.
        xyz: Point at which to calculate density. Fractional coordinates.
        atom_xyz: Atomic center. Cartesian coordinates.
        num_cells: Controls how many cells are used to estimate the sum over all unit cells. Larger values provide slightly higher accuracy but take longer."""
        zscores = (xyz + shifts - atom_xyz) / atom_rad
        # return jnp.sum(distr.prob(zscores).prod(axis=1) / distr.prob(0) ** 3)
        return jnp.sum(distr.prob(zscores).prod(axis=1))

    def parse_grid(struct: Structure) -> Float[Array, 'n3 num_el']:
        scale = struct.lattice.a
        dens = jnp.zeros((N_GRID**3, len(ELEM_VALS)))
        for site in struct.sites:
            specie = site.specie
            elem_i = ELEM_VALS.index(specie.symbol)
            atom_dens = atom_density(
                xyz, site.frac_coords, specie.average_ionic_radius * eff_scale / scale
            )
            dens = dens.at[:, elem_i].set(dens[:, elem_i] + atom_dens)
        return dens

    data_subfolder = config.data.data_folder / 'densities'

    bs = config.data.data_batch_size
    n_data = len(df.index)

    assert n_data % bs == 0

    overwrite = Confirm.ask('Overwrite previous batches?')

    # with jax.default_device(jax.devices('cpu')[0]):
    for batch in track(range(0, n_data, bs), description='Processing...'):
        fn = data_subfolder / f'batch{batch // bs}.eqx'
        if not overwrite and fn.exists():
            continue

        data = {'density': []}
        for struct in df.iloc[batch : batch + bs]['struct']:
            data['density'].append(parse_grid(struct))

        data['density'] = jnp.array(data['density'], dtype=jnp.float32)
        eqx.tree_serialise_leaves(fn, data)
