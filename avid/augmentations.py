"""Data augmentation functions."""

from functools import partial
from itertools import product
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.struct import dataclass

from avid.databatch import DataBatch

# All of the motions compatible with a cube can be described as a combined permutation of the axes
# and flipping around axes. Formation energy is not chiral, so we also include reflections and
# inversions. This means we can describe a general linear augmentation as one of 6 permutations,
# followed by one of 8 reflections (one for each axis).


@dataclass
class LinearAugmentation:
    """A linear transformation that preserves a cubic lattice."""

    perm: tuple[int]
    flips: tuple[int]

    @partial(jax.jit, static_argnums=0)
    def __call__(self, data: DataBatch) -> DataBatch:
        # B, A1, A2, A3, C
        # only the inner three get flipped
        a1, a2, a3 = self.perm
        axis_perm = (0, a1 + 1, a2 + 1, a3 + 1, 4)
        flips = [i + 1 for i in self.flips]
        new_density = jax.lax.rev(jax.lax.transpose(data.density, axis_perm), flips)
        return DataBatch(
            density=new_density,
            species=data.species,
            mask=data.mask,
            e_form=data.e_form,
            lat_abc_angles=data.lat_abc_angles,
        )

    def is_proper(self) -> bool:
        """Returns True if the augmentation preserves handedness."""
        perm_proper = self.perm in ((0, 1, 2), (1, 2, 0), (2, 0, 1))
        flip_proper = len(self.flips) % 2 == 0

        return perm_proper == flip_proper


PERMUTATIONS = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
SO3_AUGMENTATIONS = []
O3_AUGMENTATIONS = []
for perm in PERMUTATIONS:
    for flip in product((True, False), repeat=3):
        aug = LinearAugmentation(perm, tuple(i for i in (0, 1, 2) if flip[i]))
        O3_AUGMENTATIONS.append(aug)
        if aug.is_proper():
            SO3_AUGMENTATIONS.append(aug)


@dataclass
class TranslationAugmentation:
    """Applies an origin shift."""

    tau: tuple[int]

    @jax.jit
    def __call__(self, data: DataBatch) -> DataBatch:
        # B, A1, A2, A3, C
        # roll only the middle axes
        new_density = jnp.roll(data.density, shift=self.tau, axis=(1, 2, 3))
        return DataBatch(
            density=new_density,
            species=data.species,
            mask=data.mask,
            e_form=data.e_form,
            lat_abc_angles=data.lat_abc_angles,
        )


def randomly_augment(
    data: DataBatch,
    so3: bool,
    o3: bool,
    t3: bool,
    n_grid: int,
    rng: Optional[np.random.Generator] = None,
) -> nn.Module:
    """Returns a random augmentation."""
    if rng is None:
        rng = np.random.default_rng()

    if o3:
        aug = rng.choice(O3_AUGMENTATIONS)
        data = aug(data)
    elif so3:
        aug = rng.choice(SO3_AUGMENTATIONS)
        data = aug(data)

    if t3:
        tau = rng.choice(n_grid, size=3, replace=True)
        data = TranslationAugmentation(tau)(data)

    return data
