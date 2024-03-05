"""Code to load the processed data."""

from collections import defaultdict
from typing import Literal, Optional
from einops import rearrange
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import pyrallis

from avid.config import CLIConfig, DataConfig, VoxelizerConfig, LoggingLevel, MainConfig
from avid.metadata import Metadata
from avid.utils import ELEM_VALS, _debug_structure, debug_stat, debug_structure, tcheck

from jaxtyping import Float, Array, Bool, Int

n_elems = len(ELEM_VALS)

from beartype.roar import BeartypeDecorHintPep585DeprecationWarning
from warnings import filterwarnings

filterwarnings('ignore', category=BeartypeDecorHintPep585DeprecationWarning)


@tcheck
class DataBatch(eqx.Module):
    """A batch of data."""

    density: Float[Array, 'batch n_grid n_grid n_grid max_spec']
    species: Int[Array, 'batch max_spec']
    mask: Bool[Array, 'batch max_spec']
    e_form: Float[Array, 'batch']
    lat_abc_angles: Float[Array, 'batch 6']

    @classmethod
    def new_empty(batch_size: int, n_grid: int, n_spec: int):
        return DataBatch(
            jnp.empty((batch_size, n_grid, n_grid, n_grid, n_spec)),
            jnp.empty((batch_size,)),
            jnp.empty((batch_size, 6)),
        )


def load_file(config: MainConfig, file_num: int = 0):
    """Loads a file. Lacks the complex data loader logic, but easier to use for testing."""
    data_folder = config.data.data_folder
    fn = data_folder / 'densities' / f'batch{file_num}.eqx'
    num_files = len(list(data_folder.glob('densities/*.eqx')))

    data_templ = {
        'density': jnp.empty(
            (config.data.data_batch_size, config.voxelizer.n_points, n_elems),
            dtype=jnp.float32,
        ),
    }

    metadata: Metadata = eqx.tree_deserialise_leaves(
        data_folder / 'metadata.eqx', Metadata.new_empty(num_files, config.data.data_batch_size)
    )

    # data = [eqx.tree_deserialise_leaves(file, data_templ) for file in files]

    raw_data = eqx.tree_deserialise_leaves(fn, data_templ)
    data = {}
    _dens, data['species'] = jax.lax.top_k(
        raw_data['density'].max(axis=1), config.voxelizer.max_unique_species
    )
    data['density'] = jnp.permute_dims(
        jnp.take_along_axis(
            jnp.permute_dims(raw_data['density'], [0, 2, 1]), data['species'][..., None], axis=1
        ),
        [0, 2, 1],
    )
    data['species'] = data['species'].squeeze()

    assert (
        jnp.max(
            jnp.abs(
                jnp.sum(data['density'], axis=(1, 2)) - jnp.sum(raw_data['density'], axis=(1, 2))
            )
        )
        < 1e-3
    )

    nx = ny = nz = config.voxelizer.n_grid
    data['mask'] = jnp.max(jnp.abs(data['density']), axis=1) > 0
    data['density'] = rearrange(
        data['density'], 'bs (nx ny nz) nchan -> bs nx ny nz nchan', nx=nx, ny=ny, nz=nz
    )
    data['e_form'] = metadata.e_form[file_num]
    data['lat_abc_angles'] = metadata.lat_abc_angles[file_num]
    return DataBatch(**data)


def dataloader(
    config: MainConfig, split: Literal['train', 'test', 'valid'] = 'train', infinite: bool = False
):
    """Returns a generator that produces batches to train on. If infinite, repeats forever: otherwise, stops when all data has been yielded."""
    ngrid = config.voxelizer.n_grid
    data_folder = config.data.data_folder
    files = sorted(list(data_folder.glob('densities/*.eqx')))

    splits = np.cumsum([config.data.train_split, config.data.valid_split, config.data.test_split])
    total = splits[-1]
    split_inds = np.zeros(total)
    split_inds[: splits[0]] = 0
    split_inds[splits[0] : splits[1]] = 1
    split_inds[splits[1] :] = 2

    split_i = ['train', 'valid', 'test'].index(split)

    # data = [eqx.tree_deserialise_leaves(file, data_templ) for file in files]

    split_idx = np.arange(len(files))
    split_idx = split_idx[split_inds[split_idx % total] == split_i]

    while True:
        batch_inds = np.split(
            np.random.permutation(split_idx), len(split_idx) // config.train_batch_multiple
        )
        for batch in batch_inds:
            batch_data = [load_file(config, i) for i in batch]

            yield jax.tree_map(lambda *args: jnp.concat(args, axis=0), *batch_data)

        if not infinite:
            break


def num_elements_class(batch):
    # 2, 3, 4, 5 are values
    # map to 0, 1, 2, 3
    return (
        jax.nn.one_hot(batch['species'], jnp.max(batch['species']).item(), dtype=jnp.int16)
        .max(axis=1)
        .sum(axis=1)
    ) - 2


if __name__ == '__main__':
    config = pyrallis.parse(config_class=MainConfig)
    config.cli.set_up_logging()

    f1 = load_file(config, 300)
    debug_structure(conf=f1)
    jax.debug.print('Batch: {}', f1)

    debug_structure(conf=next(dataloader(config)))
    jax.debug.print('Batch: {}', next(dataloader(config)))
