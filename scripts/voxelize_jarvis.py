from jaxtyping import Array, Float, PRNGKeyArray, Float32, Int
import jax
import jax.numpy as jnp
import functools as ft
from einops import rearrange, reduce
import flax
import flax.linen as nn
import numpy as np
import pandas as pd
clean = pd.read_pickle('precomputed/jarvis_dft3d_cleaned/dataframe.pkl')
clean = clean.sort_index()
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

elements = set()
for struct in clean['atoms']:
    elements.update(set(struct.elements))

# print(len(elements))
# print(elements)
# print({e.symbol: (e.average_ionic_radius, e.atomic_radius) for e in elements})

sorted_elements = sorted(elements, key=lambda e: e.number)
elem_vals = [elem.symbol for elem in sorted_elements]

dist = tfd.GeneralizedNormal(0.5, 2, 2)

N_GRID = 24
num_cells=4
grid_vals = jnp.linspace(0, 1, N_GRID + 1)[:-1]
xx, yy, zz = jnp.meshgrid(grid_vals, grid_vals, grid_vals)
xyz = rearrange(jnp.array([xx, yy, zz]), 'd n1 n2 n3 -> (n1 n2 n3) d')
xx, yy, zz = jnp.meshgrid(*[jnp.arange(-num_cells, num_cells + 1, dtype=jnp.float32)
                            for _ in range(3)])
shifts = rearrange(jnp.array([xx, yy, zz]), 'd n1 n2 n3 -> (n1 n2 n3) d')
dist.prob(xyz[None, :, :] + shifts[:, None, :]).prod(axis=2).sum(axis=0).mean()


from pymatgen.core import Structure
N_GRID = 24
# https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.108.058301
power = 2.4
eff_scale = jnp.array(0.7)
num_cells = 4


distr = tfd.GeneralizedNormal(0, 1, power)
grid_vals = jnp.linspace(0, 1, N_GRID + 1)[:-1]
xx, yy, zz = jnp.meshgrid(grid_vals, grid_vals, grid_vals)
xyz = rearrange(jnp.array([xx, yy, zz]), 'd n1 n2 n3 -> (n1 n2 n3) d')

@ft.partial(jax.vmap, in_axes=(0, None, None, None))
@ft.partial(jax.jit, static_argnames=['distr', 'num_cells'])
def atom_density(xyz: Float[Array, '3'], atom_xyz: Float[Array, '3'], atom_rad: Float[Array, ''],num_cells: int = 4, distr: tfd.Distribution = distr) -> Float[Array, '']:
    """Atom density.
    xyz: Point at which to calculate density. Fractional coordinates.
    atom_xyz: Atomic center. Cartesian coordinates.
    num_cells: Controls how many cells are used to estimate the sum over all unit cells. Larger values provide slightly higher accuracy but take longer."""
    xx, yy, zz = jnp.meshgrid(*[jnp.arange(-num_cells, num_cells + 1, dtype=jnp.float32)
                                for _ in range(3)])
    shifts = rearrange(jnp.array([xx, yy, zz]), 'd n1 n2 n3 -> (n1 n2 n3) d')
    zscores = (xyz + shifts - atom_xyz) / atom_rad
    return jnp.sum(distr.prob(zscores).prod(axis=1) / distr.prob(0) ** 3)


def parse_grid(struct: Structure) -> (Float[Array, 'n3'], Int[Array, 'n3']):
    scale = struct.lattice.a
    total_dens = 0
    dens = jnp.zeros((N_GRID ** 3, len(elem_vals)))
    for site in struct.sites:
        specie = site.specie
        elem_i = elem_vals.index(specie.symbol)
        radius = specie.average_ionic_radius
        if radius == 0:
            radius = specie.atomic_radius
        if radius == 0:
            raise ValueError
        atom_dens = atom_density(xyz, site.frac_coords, radius * eff_scale / scale, num_cells)
        dens = dens.at[:, elem_i].set(dens[:, elem_i] + atom_dens)

    total_dens = dens.sum(axis=1)
    species = dens.argmax(axis=1)
    return (dens, total_dens, species)


from tqdm import tqdm
from flax.serialization import msgpack_serialize, msgpack_restore
import gc

make_data = lambda: {
    'density': [],
    'species': [],
    'mask': [],
    'space_group': [],
    'e_form': [],
    'bandgap': [],
    'e_total': [],
    'e_hull': [],
    'magmom': [],
    'cell_density': [],
    'index': []
}

data = make_data()

BATCH_SIZE = 32

rng = np.random.Generator(np.random.PCG64(29205))
clean = clean.loc[rng.permutation(clean.index)]

if __name__ == '__main__':
    from rich.prompt import Confirm
    if not Confirm.ask('Regenerate voxelized files?'):
        raise ValueError('Aborted')

    with jax.default_device(jax.devices('cuda')[1]):
        for i, idx in tqdm(list(enumerate(clean.index))):
            row = clean.loc[idx]
            struct = row['atoms']
            dens, _sum_dens, _spec = parse_grid(struct)
            mask, species = jax.lax.top_k(dens.max(axis=0), max(clean['num_spec']))
            mask = mask > 0
            data['density'].append(dens[:, species].reshape(N_GRID, N_GRID, N_GRID, -1))
            data['species'].append(species)
            data['mask'].append(mask)
            data['cell_density'].append(row['density'])
            data['e_hull'].append(row['ehull'])
            data['index'].append(idx)
            for key in data.keys():
                if key not in ['density'] and key in row.index:
                    data[key].append(row[key])

            if (i + 1) % BATCH_SIZE == 0:
                for col in ('species', 'space_group'):
                    data[col] = jnp.array(data[col], dtype=jnp.uint8)
                for col in ('index',):
                    data[col] = jnp.array(data[col], dtype=jnp.uint32)
                for col in ('mask',):
                    data[col] = jnp.array(data[col], dtype=jnp.bool)
                for col in ('density', 'e_form', 'bandgap', 'e_total', 'e_hull', 'magmom', 'cell_density'):
                    data[col] = jnp.array(data[col], dtype=jnp.float32)

                batch_ind = (i + 1) // BATCH_SIZE - 1

                with open(f'precomputed/jarvis_dft3d_cleaned/densities/batch{batch_ind}.mpk', 'wb') as out:
                    out.write(msgpack_serialize(data))

                del data
                data = make_data()