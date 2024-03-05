"""Utilities."""

from abc import ABCMeta
from json import dumps
import logging
import jax.numpy as jnp
import jax
import numpy as np
import equinox as eqx
from pymatgen.core import Element
from dataclasses import asdict, is_dataclass
from rich.pretty import pprint
from rich import print_json, print
from rich.tree import Tree
from rich.style import Style


ELEM_VALS = 'Li Be B N O F Na Mg Al Si S K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Rb Sr Y Zr Nb Mo Ru Rh Pd Ag Cd In Sn Sb Te Cs Ba La Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi'.split(
    ' '
)


def val_to_elem(val: int) -> str:
    return ELEM_VALS[val]


def elem_to_val(elem: str | Element) -> int:
    if isinstance(elem, Element):
        elem = elem.symbol

    return ELEM_VALS.index(elem)


class AbstractTreeVisitor(metaclass=ABCMeta):
    def __init__(self):
        pass

    def jax_arr(self, arr: jax.Array):
        raise NotImplementedError()

    def scalar(self, x: int | float):
        raise NotImplementedError()

    def np_arr(self, arr: np.ndarray):
        raise NotImplementedError()


class StatVisitor(AbstractTreeVisitor):
    def __init__(self) -> None:
        super().__init__()

    def jax_arr(self, arr: jax.Array):
        flat = arr.flatten().astype(jnp.float32)
        inds = 1 + 0.01 * jnp.cos(jnp.arange(len(flat), dtype=jnp.float32))
        return f'{(flat * inds).mean().item():.3f}'

    def scalar(self, x: int | float):
        return f'{x:.2f}' + x.__class__.__name__[0]

    def np_arr(self, arr: np.ndarray):
        return self.jax_arr(jnp.array(arr))


class StructureVisitor(AbstractTreeVisitor):
    def __init__(self):
        super().__init__()

    def jax_arr(self, arr: jax.Array):
        return f'{arr.dtype}{list(arr.shape)}'

    def scalar(self, x: int | float):
        return f'{type(x)}=' + str(x)[:3]

    def np_arr(self, arr: np.ndarray):
        return f'np{list(arr.shape)}'


COLORS = [
    '#00a0ec',
    '#00bc70',
    '#deca00',
    '#ff7300',
    '#d83990',
    '#7555d3',
    '#8ac7ff',
    '#00f0ff',
    '#387200',
    '#aa2e00',
    '#ff7dc6',
    '#e960ff',
]

KWARGS = [dict(color=color) for color in COLORS]
KWARGS[0]['bold'] = True

STYLES = [Style(**kwargs) for kwargs in KWARGS]


def tree_from_dict(base, obj, depth=0):
    style = STYLES[depth % len(STYLES)]
    if isinstance(obj, dict):
        for k, v in obj.items():
            child = base.add(k, style=style)
            tree_from_dict(child, v, depth + 1)
    else:
        base.add(obj, style=style)


def tree_traverse(visitor: AbstractTreeVisitor, obj, max_depth=2):
    if isinstance(obj, jax.Array):
        return visitor.jax_arr(obj)
    elif isinstance(obj, np.ndarray):
        return visitor.np_arr(obj)
    elif isinstance(obj, (list, tuple)):
        if max_depth == 0:
            return '[...]'
        else:
            return [tree_traverse(visitor, child, max_depth - 1) for child in obj]
    elif isinstance(obj, (float, int)):
        return visitor.scalar(obj)
    elif isinstance(obj, dict):
        if max_depth == 0:
            return '{...}'
        else:
            return {k: tree_traverse(visitor, v, max_depth - 1) for k, v in obj.items()}
    elif isinstance(obj, eqx.Module) or is_dataclass(obj):
        return {obj.__class__.__name__: tree_traverse(visitor, asdict(obj), max_depth)}
    else:
        return obj.__class__.__name__


def show_obj(obj):
    # print_json(data=obj)
    for k, v in obj.items():
        tree = Tree(label=k, style=STYLES[0])
        tree_from_dict(tree, v, depth=1)
        print(tree)


def _debug_structure(tree_depth=2, **kwargs):
    """Prints out the structure of the inputs."""
    show_obj(
        {
            f'{k} structure': tree_traverse(StructureVisitor(), v, tree_depth)
            for k, v in kwargs.items()
        }
    )


def _debug_stat(tree_depth=2, **kwargs):
    """Prints out a reduction of the inputs. Is almost the mean, but with a small fudge factor so differently-shaped arrays will have different summaries."""
    show_obj({f'{k} stat': tree_traverse(StatVisitor(), v, tree_depth) for k, v in kwargs.items()})


def debug_structure(**kwargs):
    """Prints out the structure of the inputs."""
    jax.debug.callback(_debug_structure, **kwargs)


def debug_stat(**kwargs):
    """Prints out a reduction of the inputs. Is almost the mean, but with a small fudge factor so differently-shaped arrays will have different summaries."""
    jax.debug.callback(_debug_stat, **kwargs)
