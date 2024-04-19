"""File to serialize equivariant basis for different-size cubes."""

import jax


from emlp.reps import V,sparsify_basis,T
import emlp.groups as groups
import emlp
import jax.numpy as jnp
import numpy as np
from rich.prompt import Confirm, IntPrompt
from emlp.reps.linear_operators import LazyPerm
from einops import rearrange
from math import comb
from itertools import product
import pandas as pd


def gen_elements(n, k, d):
    poly = jnp.array([n ** i for i in range(k - 1)], dtype=jnp.int32)
    pows = jnp.arange(k - 1, dtype=jnp.int32)[::-1]
    parity = 1 if n % 2 == 1 else 2
    coefs = jnp.dot(parity ** pows, poly)
    dom = ((n - parity) * coefs) // 2 + parity ** (k - 1)
    return comb(dom + (d - 1), d)


class SquarePerm(emlp.Group):
    def __init__(self, n: int, shift_subgroup: int = 1):
        assert n % shift_subgroup == 0

        base = jnp.arange(n * n).reshape(n, n)

        rots = []
        taus = []
        for ax in range(2):
            rots.append(jnp.flip(base, axis=ax))
            taus.append(jnp.roll(base, shift_subgroup, axis=ax))

        rots.append(jnp.flip(jnp.transpose(base), axis=0))
        rots.append(jnp.transpose(base))
        rots.append(jnp.flip(base, (0, 1)))

        perms = rearrange(jnp.array([rots + taus]), '1 b nx ny -> b (nx ny)')
        self.discrete_generators = [LazyPerm(perm) for perm in perms]
        super().__init__(n)

class CubicPerm(emlp.Group):
    def __init__(self, n: int, shift_subgroup: int = 1):
        assert n % shift_subgroup == 0

        base = jnp.arange(n * n * n).reshape(n, n, n)

        rots = []
        flips = []
        taus = []
        for ax in range(3):
            if ax > 0:
                axes = list(range(3))
                axes[0], axes[ax] = axes[ax], axes[0]
                rots.append(jnp.permute_dims(base, axes=axes))
            flips.append(jnp.flip(base, ax))
            taus.append(jnp.roll(base, shift_subgroup, axis=ax))

        perms = rearrange(jnp.array([rots + flips + taus]), '1 b nx ny nz -> b (nx ny nz)')
        self.discrete_generators = [LazyPerm(perm) for perm in perms]
        super().__init__(n)

Sq = SquarePerm
Cu = CubicPerm

if __name__ == '__main__':
    if Confirm.ask('Solve for equivariant basis?'):
        n = IntPrompt.ask('Input number of voxels per cube side:')
        print(f'Generating subspace with {gen_elements(n, k=2, d=3)} elements')
        B = (T(1) >> T(1))(Cu(n)).equivariant_basis()
        fn = f'precomputed/equiv_basis_{n}.npy'
        jnp.save(fn, np.array(B))
        print('Saved to', fn)
    else:
        print('Aborted')