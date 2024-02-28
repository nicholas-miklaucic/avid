"""Code to load the processed data."""
from typing import Literal
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import pyrallis

from avid.config import CLIConfig, DataConfig, DataEncoderConfig, LoggingLevel, MainConfig
from avid.utils import _debug_structure, debug_structure


def dataloader(config: MainConfig, split: Literal['train', 'test', 'valid'] = 'train'):
    """Returns a generator that produces batches to train on."""
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
            (config.data.data_batch_size, config.data_encoder.n_points), dtype=jnp.float32
        ),
        'species': jnp.empty(
            (config.data.data_batch_size, config.data_encoder.n_points), dtype=jnp.int16
        ),
    }

    e_form_templ = jnp.empty(len(files) * config.data.data_batch_size, dtype=jnp.float32)
    e_forms = eqx.tree_deserialise_leaves(data_folder / 'e_forms.eqx', e_form_templ)
    e_forms = e_forms.reshape(config.data.data_batch_size, len(files))

    # data = [eqx.tree_deserialise_leaves(file, data_templ) for file in files]

    split_idx = np.arange(len(files))
    split_idx = split_idx[split_inds[split_idx % total] == split_i]

    while True:
        batch_inds = np.split(
            np.random.permutation(split_idx), len(split_idx) // config.train_batch_multiple
        )
        for batch in batch_inds:
            data_batch = {'density': [], 'species': [], 'e_form': []}
            for i in batch:
                data = eqx.tree_deserialise_leaves(files[i], data_templ)
                for key in data_batch:
                    if key in data:
                        data_batch[key].append(data[key])
                    elif key == 'e_form':
                        data_batch[key].append(e_forms[:, i].reshape(-1, 1))

            for key in data_batch:
                data_batch[key] = jnp.vstack(data_batch[key])

            data_batch['n_elements_label'] = num_elements_class(data_batch)

            yield data_batch

        if split != 'train':
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
