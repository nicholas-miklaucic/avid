"""Code to load the processed data."""

import os

os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pyrallis
from einops import rearrange

from avid.config import MainConfig
from avid.databatch import DataBatch
from avid.metadata import Metadata
from avid.utils import ELEM_VALS, debug_structure

n_elems = len(ELEM_VALS)

from warnings import filterwarnings

from beartype.roar import BeartypeDecorHintPep585DeprecationWarning

filterwarnings('ignore', category=BeartypeDecorHintPep585DeprecationWarning)


def load_file(config: MainConfig, file_num=0):
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


def dataloader_base(
    config: MainConfig, split: Literal['train', 'test', 'valid'] = 'train', infinite: bool = False
):
    """Returns a generator that produces batches to train on. If infinite, repeats forever: otherwise, stops when all data has been yielded."""
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

    yield len(split_idx) // config.train_batch_multiple

    split_files = np.array([None for _ in range(len(files))])

    batch_inds = np.split(
        np.random.permutation(split_idx),
        len(split_idx) // config.train_batch_multiple,
    )

    # debug_structure(splidx=split_idx, bi0=batch_inds[0], bi=batch_inds)

    device = config.device.jax_device

    if isinstance(device, jax.sharding.PositionalSharding):
        # rearrange to fit shape of databatch
        device = device.reshape(-1, 1, 1, 1, 1)

    with jax.default_device(jax.devices('cpu')[0]):
        for batch in batch_inds:
            split_files[batch] = [load_file(config, i) for i in batch]

            batch_data = split_files[batch]
            collated = jax.tree_map(lambda *args: jnp.concat(args, axis=0), *batch_data)
            yield collated.device_put(device)

    while infinite:
        batch_inds = np.split(
            np.random.permutation(split_idx),
            len(split_idx) // config.train_batch_multiple,
        )
        for batch in batch_inds:
            batch_data = [split_files[i] for i in batch]
            collated = jax.tree_map(lambda *args: jnp.concat(args, axis=0), *batch_data)
            yield collated.device_put(device)


def dataloader(
    config: MainConfig, split: Literal['train', 'test', 'valid'] = 'train', infinite: bool = False
):
    dl = dataloader_base(config, split, infinite)
    steps_per_epoch = next(dl)
    return (steps_per_epoch, dl)


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

    from tqdm import tqdm

    steps_per_epoch, dl = dataloader(config, split='valid', infinite=True)

    means = []
    for _i in tqdm(np.arange(steps_per_epoch * 2)):
        batch = next(dl)
        means.append(batch.density.mean())

    print(jnp.mean(jnp.array(means)))

    debug_structure(conf=next(dl))
