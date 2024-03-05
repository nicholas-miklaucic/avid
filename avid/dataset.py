"""Code to load the processed data."""

from typing import Literal
from einops import rearrange
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import pyrallis

from avid.config import CLIConfig, DataConfig, VoxelizerConfig, LoggingLevel, MainConfig
from avid.metadata import Metadata
from avid.utils import ELEM_VALS, _debug_structure, debug_structure

from jaxtyping import Float, Array

n_elems = len(ELEM_VALS)


class DataBatch(eqx.Module):
    """A batch of data."""

    density: Float[Array, 'batch n_grid n_grid n_grid n_spec']
    e_form: Float[Array, 'batch']
    lat_abc_angles: Float[Array, 'batch 6']

    @classmethod
    def new_empty(batch_size: int, n_grid: int, n_spec: int):
        return DataBatch(
            jnp.empty((batch_size, n_grid, n_grid, n_grid, n_spec)),
            jnp.empty((batch_size,)),
            jnp.empty((batch_size, 6)),
        )


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

    data_templ = {
        'density': jnp.empty(
            (config.data.data_batch_size, config.voxelizer.n_points, n_elems),
            dtype=jnp.float32,
        ),
    }

    metadata: Metadata = eqx.tree_deserialise_leaves(
        data_folder / 'metadata.eqx', Metadata.new_empty(len(files), config.data.data_batch_size)
    )

    # data = [eqx.tree_deserialise_leaves(file, data_templ) for file in files]

    split_idx = np.arange(len(files))
    split_idx = split_idx[split_inds[split_idx % total] == split_i]

    while True:
        batch_inds = np.split(
            np.random.permutation(split_idx), len(split_idx) // config.train_batch_multiple
        )
        for batch in batch_inds:
            data_batch = {'density': []}
            for i in batch:
                data = eqx.tree_deserialise_leaves(files[i], data_templ)
                data_batch['density'].append(
                    rearrange(
                        data['density'],
                        'bs (nx ny nz) nspec -> bs nx ny nz nspec',
                        bs=config.data.data_batch_size,
                        nx=ngrid,
                        ny=ngrid,
                        nz=ngrid,
                        nspec=n_elems,
                    )
                )

            for key in data_batch:
                data_batch[key] = jnp.vstack(data_batch[key])

            data_batch['e_form'] = metadata.e_form[jnp.array(batch)]
            data_batch['lat_abc_angles'] = metadata.lat_abc_angles[jnp.array(batch)]

            for meta_key in ('e_form', 'lat_abc_angles'):
                data_batch[meta_key] = jnp.concatenate(data_batch[meta_key], axis=0)

            yield DataBatch(**data_batch)

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

    debug_structure(conf=next(dataloader(config)))
    jax.debug.print('Batch: {}', next(dataloader(config)))
