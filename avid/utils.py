"""Utilities."""

import logging
import jax.numpy as jnp
import jax
import numpy as np
from pymatgen.core import Element


ELEM_VALS = 'Li Be B N O F Na Mg Al Si S K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Rb Sr Y Zr Nb Mo Ru Rh Pd Ag Cd In Sn Sb Te Cs Ba La Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi'.split(
    ' '
)


def val_to_elem(val: int) -> str:
    return ELEM_VALS[val]


def elem_to_val(elem: str | Element) -> int:
    if isinstance(elem, Element):
        elem = elem.symbol

    return ELEM_VALS.index(elem)


def structure(obj):
    if isinstance(obj, jax.Array):
        return f'{obj.dtype}{obj.shape}'
    elif isinstance(obj, np.ndarray):
        return list(obj.shape)
    elif isinstance(obj, (list, tuple)):
        return list(map(structure, obj))
    elif isinstance(obj, (float, int)):
        return 'scalar'
    elif isinstance(obj, dict):
        return {k: structure(v) for k, v in obj.items()}
    else:
        return str(type(obj))


def summary_stat(obj):
    if isinstance(obj, jax.Array):
        flat = obj.flatten()
        inds = 1 + 0.01 * jnp.cos(jnp.arange(len(flat), dtype=obj.dtype))
        return (flat * inds).mean().item()
    elif isinstance(obj, np.ndarray):
        return summary_stat(jnp.array(obj))
    elif isinstance(obj, (list, tuple)):
        return list(map(structure, obj))
    elif isinstance(obj, (float, int)):
        return f'scalar={obj}'
    elif isinstance(obj, dict):
        return {k: summary_stat(v) for k, v in obj.items()}
    else:
        return str(type(obj))


def _debug_structure(**kwargs):
    """Prints out the structure of the inputs."""
    for k, v in kwargs.items():
        logging.info(f'{k:>30} structure:\t{structure(v)}')


def _debug_stat(**kwargs):
    """Prints out a reduction of the inputs. Is almost the mean, but with a small fudge factor so differently-shaped arrays will have different summaries."""
    for k, v in kwargs.items():
        for k, v in kwargs.items():
            logging.info(f'{k:>30} stat:\t{summary_stat(v)}')


def debug_structure(**kwargs):
    """Prints out the structure of the inputs."""
    jax.debug.callback(_debug_structure, **kwargs)


def debug_stat(**kwargs):
    """Prints out a reduction of the inputs. Is almost the mean, but with a small fudge factor so differently-shaped arrays will have different summaries."""
    jax.debug.callback(_debug_stat, **kwargs)
