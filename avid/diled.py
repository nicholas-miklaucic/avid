"""
Generalizable Diffusion with Learned Encoding-Decoding.
"""

from typing import Sequence

import einops
import jax
import jax.numpy as jnp
import numpy as np
from chex import dataclass
from eins import EinsOp
from flax import linen as nn

from avid.databatch import DataBatch
from avid.diffusion import DiffusionInput, DiffusionModel
from avid.layers import LazyInMLP

# class O3ImageEmbed(nn.Module):
#     """Embeds a 3D image into a sequnce of tokens."""

#     patch_size: int
#     patch_latent_dim: int
#     patch_heads: int
#     pos_embed: nn.Module

#     @tcheck
#     @nn.compact
#     def __call__(self, im: Float[Array, 'I I I C']) -> Float[Array, '_ip3 _dimout']:
#         i = im.shape[0]
#         p = self.patch_size
#         c = im.shape[-1]
#         assert i % p == 0
#         n = i // p

#         patches = im.reshape(n, p, n, p, n, p, c)
#         patches = rearrange(patches, 'n1 p1 n2 p2 n3 p3 c -> (n1 n2 n3) p1 p2 p3 c')

#         embed = jax.vmap(O3Patchify(self.patch_latent_dim, inner_dim=self.patch_heads))(patches)
#         # legendre
#         # pos_embed = self.pos_embed(rearrange(embed, '(i1 i2 i3) c -> i1 i2 i3 c', i1=n, i2=n,
#         # i3=n))
#         # return rearrange(pos_embed, 'i1 i2 i3 c -> (i1 i2 i3) c', i1=n, i2=n, i3=n)

#         # learned
#         pos_embed = self.pos_embed(embed)
#         return pos_embed


def conv_basis(conv_size: int) -> np.ndarray:
    conv_basis = np.zeros((conv_size, conv_size, conv_size), dtype=np.int_)
    n = (conv_size + 2) // 2
    i = 0
    for xyz in np.mgrid[0:n, 0:n, 0:n].reshape(3, -1).T:
        if not np.allclose(np.argsort(xyz), np.arange(3)):
            continue
        for s_xyz in np.mgrid[0:2, 0:2, 0:2].reshape(3, -1).T:
            # flip
            flip_xyz = xyz - 2 * xyz * s_xyz
            for perm in [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]:
                # move -n - n into 0 - conv_size
                xi, yi, zi = flip_xyz[perm] + (n - 1)
                conv_basis[xi, yi, zi] = i
        i += 1
    return conv_basis


class SpeciesTwoWayEmbed(nn.Module):
    """Embedding module that supports recovering densities."""
    n_species: int
    embed_dim: int

    def species_embed_matrix(self):
        """Generates species embedding matrix of orthogonal vectors: (n_species, embed_dim)."""
        pass

    def __call__(self, data: DataBatch):
        """Embeds the batch as a 3D image, shape batch n n n embed_dim."""
        pass

    def decode(self, im, data: DataBatch):
        """Projects inputs to obtain the original densities."""
        pass

class OrthoSpeciesEmbed(SpeciesTwoWayEmbed):
    """Embedding module that supports recovering densities. Uses orthogonal embeddings."""
    n_species: int
    embed_dim: int

    def setup(self):
        self.mat = jnp.eye(self.embed_dim, self.n_species)
        self.ii, self.jj = jnp.tril_indices_from(self.mat, k=-1)
        self.vs = self.param(
            'embed_raw',
            nn.initializers.truncated_normal(stddev=0.2, dtype=jnp.float32, lower=-2, upper=2),
            (self.n_species, self.embed_dim),
        )

    def species_embed_matrix(self):
        """Generates species embedding matrix of orthogonal vectors: (n_species, embed_dim)."""
        vvt = EinsOp('m n1, m n2 -> m n1 n2')(self.vs, self.vs)
        vtv = EinsOp('m n, m n -> m 1 1')(self.vs, self.vs)
        mats = jnp.eye(self.embed_dim, self.embed_dim) - 2 / vtv * vvt
        orth = jnp.linalg.multi_dot(mats)
        return orth[: self.n_species]

    def __call__(self, data: DataBatch):
        """Embeds the batch as a 3D image, shape batch n n n embed_dim."""
        spec = jax.nn.one_hot(data.species, self.n_species, dtype=jnp.float32)
        # debug_structure(spec=spec, mat=self.species_embed_matrix())
        return einops.einsum(
            data.density,
            spec,
            self.species_embed_matrix(),
            'batch n1 n2 n3 max_spec, batch max_spec n_spec, n_spec emb -> batch n1 n2 n3 emb',
        )

    def decode(self, im, data: DataBatch):
        """Projects inputs to obtain the original densities."""
        proj = EinsOp('batch n n n emb, n_spec emb -> batch n n n n_spec')(
            im, self.species_embed_matrix()
        )
        return proj
    
class LossySpeciesEmbed(SpeciesTwoWayEmbed):
    """Embedding module that supports recovering densities."""
    n_species: int
    embed_dim: int

    def setup(self):
        self.emb = nn.Embed(self.n_species, self.embed_dim, dtype=jnp.bfloat16)
        self.decoder = LazyInMLP(
            inner_dims=[64],
            out_dim=self.n_species,
            name='species_decoder',
            inner_act=nn.gelu,            
        )

    def species_embed_matrix(self):
        """Generates species embedding matrix of orthogonal vectors: (n_species, embed_dim)."""
        return self.emb.embedding

    def __call__(self, data: DataBatch):
        """Embeds the batch as a 3D image, shape batch n n n embed_dim."""
        spec = jax.nn.one_hot(data.species, self.n_species, dtype=jnp.bfloat16)
        # debug_structure(spec=spec, mat=self.species_embed_matrix())
        embs = self.emb(data.species) * data.mask[..., None]        
        return einops.einsum(
            data.density,
            embs,
            'batch n1 n2 n3 max_spec, batch max_spec emb -> batch n1 n2 n3 emb',
        )

    def decode(self, im, data: DataBatch):
        """Projects inputs to obtain the original densities."""

        out = self.decoder(im, training=True)
        out = jnp.take_along_axis(out, data.species[:, None, None, None, :], axis=-1)
        return out


class GConv(nn.Module):
    """Group-equivariant convolution."""

    conv_size: int
    in_features: int
    out_features: int
    stride: int
    transpose: bool = False

    def setup(self):
        self.basis = conv_basis(self.conv_size)
        self.basis = [np.equal(self.basis, k) for k in range(max(self.basis.flatten()) + 1)]
        self.basis = jnp.array(self.basis, dtype=jnp.float32)
        self.subspace_dim = self.basis.shape[0]
        self.kernel = self.param(
            'kernel',
            nn.initializers.normal(stddev=1e-1),
            (self.subspace_dim, self.in_features, self.out_features),
        )

        self.project = EinsOp('subspace c_in c_out, subspace n1 n2 n3 -> n1 n2 n3 c_in c_out')

        self.conv = nn.Conv(
            features=self.out_features,
            kernel_size=(self.conv_size, self.conv_size, self.conv_size),
            use_bias=False,
            padding='CIRCULAR',
            dtype=jnp.bfloat16,
            strides=self.stride,
        )

        self.conv_transpose = nn.ConvTranspose(
            features=self.in_features,
            kernel_size=(self.conv_size, self.conv_size, self.conv_size),
            use_bias=False,
            padding='CIRCULAR',
            dtype=jnp.bfloat16,
            strides=[self.stride, self.stride, self.stride],
            transpose_kernel=True,
        )

    def __call__(self, x):
        if self.transpose:
            conv = self.conv_transpose
        else:
            conv = self.conv

        k_proj = self.project(self.kernel, self.basis).astype(jnp.bfloat16)

        conv_out = conv.apply({'params': {'kernel': k_proj}}, x)
        return conv_out


class EncoderDecoder(nn.Module):
    """Embeds species as orthonormal vectors and then applies a patch embedding with learned linear decoder."""

    patch_latent_dim: int
    patch_conv_sizes: Sequence[int]
    patch_conv_strides: Sequence[int]
    patch_conv_features: Sequence[int]
    use_dec_conv: bool
    species_embed_dim: int
    n_species: int

    def setup(self):
        self.spec_emb = LossySpeciesEmbed(
            n_species=self.n_species, embed_dim=self.species_embed_dim
        )
        prev_feats = self.species_embed_dim
        enc_convs = []
        dec_convs = []
        for size, stride, feats in zip(
            self.patch_conv_sizes, self.patch_conv_strides, self.patch_conv_features
        ):
            kwargs = dict(conv_size=size, in_features=prev_feats, out_features=feats, stride=stride)
            for convs, transpose in zip((enc_convs, dec_convs), (False, True)):
                if transpose and not self.use_dec_conv:
                    continue
                block = [
                    GConv(transpose=transpose, **kwargs),
                    nn.LayerNorm(dtype=jnp.bfloat16),
                ]
                convs.append(nn.Sequential(block))
            prev_feats = feats

        self.encoder_conv = nn.Sequential(enc_convs)
        if self.use_dec_conv:
            self.decoder_conv = nn.Sequential(dec_convs[::-1])

        self.patch_proj = nn.DenseGeneral(
            features=self.patch_latent_dim, axis=-1, dtype=jnp.bfloat16
        )

        if self.use_dec_conv:
            self.dec_patch_proj = nn.DenseGeneral(features=feats, axis=-1, dtype=jnp.bfloat16)
        else:
            self.dec_patch_proj = nn.DenseGeneral(
                features=np.prod(self.patch_conv_strides) ** 3 * self.species_embed_dim,
                axis=-1,
                dtype=jnp.bfloat16,
            )

    def encode(self, data: DataBatch, training: bool):
        spec_emb = self.spec_emb(data)
        enc = self.encoder_conv(spec_emb)
        patches = self.patch_proj(enc)
        return (spec_emb, enc, patches)

    def patch_decode(self, patches, data: DataBatch, training: bool):
        dec = self.dec_patch_proj(patches)
        if self.use_dec_conv:
            spec_emb = self.decoder_conv(dec)
        else:
            spec_emb = EinsOp(
                'b i i i (p p p d) -> b (i p) (i p) (i p) d',
                symbol_values={'d': self.species_embed_dim},
            )(dec)

        dens = self.spec_emb.decode(spec_emb, data)
        return (dens, spec_emb, dec)


class Category:
    """Get category from data."""

    def __call__(self, data: DataBatch):
        """Returns list of integers for category."""
        raise NotImplementedError

    @property
    def num_categories(self) -> int:
        raise NotImplementedError


@dataclass
class EFormCategory(Category):
    num_cats: int = 8
    min_bin: float = -0.5
    max_bin: float = 5.0

    @property
    def bins(self):
        return jnp.linspace(self.min_bin, self.max_bin, self.num_categories - 2)

    def __call__(self, data: DataBatch):
        return jnp.digitize(data.e_form, self.bins)

    @property
    def num_categories(self) -> int:
        return self.num_cats
    

@dataclass
class SpaceGroupCategory(Category):
    just_cubic: bool = True

    def __call__(self, data: DataBatch):
        if self.just_cubic:
            return data.space_group - 195
        else:
            return data.space_group - 1
    
    @property
    def num_categories(self) -> int:
        if self.just_cubic:
            return 230
        else:
            return 230 - 194


class DiLED(nn.Module):
    """DiLED model."""

    encoder_decoder: EncoderDecoder
    diffusion: DiffusionModel
    category: Category
    w: float = 1
    class_dropout: float = 0.5

    @nn.compact
    def __call__(self, data: DataBatch, training: bool):
        """Runs a training step, returning loss. Uses time, noise RNG keys."""
        x_0, conv, patch = self.encoder_decoder.encode(data, training=training)
        enc = patch
        beta_0 = 1e-3
        eps_0 = jax.random.normal(self.make_rng('noise'), shape=enc.shape)
        x_1 = enc + beta_0 * eps_0

        dens, rec_x_0, dec_conv = self.encoder_decoder.patch_decode(x_1, data, training=training)

        def voxel_loss(x1, x2):
            return jnp.sqrt(jnp.mean(jnp.square(x1 - x2)))

        def l_rec():
            dens_aligned = jnp.take_along_axis(dens, data.species[:, None, None, None, :], axis=-1)
            rec_loss = voxel_loss(dens_aligned, data.density)
            loss = rec_loss + 0.01 * voxel_loss(jnp.zeros_like(patch), patch)
            return loss * 1000

        if self.w == 0:
            loss = l_rec()
            return (loss, {'loss': loss, 'rec_loss': loss, 'diffuser_loss': 0, 'align_loss': 0})

        t = jax.random.randint(
            self.make_rng('time'), [1], minval=1, maxval=self.diffusion.schedule.max_t
        )
        y = self.category(data)

        y_mask = jnp.where(
            jax.random.uniform(self.make_rng('noise'), (y.shape[0],)) < self.class_dropout,
            -jnp.ones_like(y),
            y,
        )
        eps = jax.random.normal(self.make_rng('noise'), shape=x_1.shape)

        def l_align(eps):
            x_2 = self.diffusion.q_sample(x_1, 2, eps)
            mu_2_1 = self.diffusion.mu_eps_sigma(
                DiffusionInput(x_t=x_2, t=1, y=y_mask), training=training
            )
            loss = voxel_loss(patch.astype(jnp.float32), mu_2_1['mu'].astype(jnp.float32))
            return loss * 1000

        def l_diffusion(eps):
            x_t = self.diffusion.q_sample(enc, t, eps)
            mu_eps = self.diffusion.mu_eps_sigma(
                DiffusionInput(x_t=x_t, t=t, y=y_mask), training=training
            )
            return voxel_loss(mu_eps['eps'].astype(jnp.float32), eps.astype(jnp.float32))

        rec_loss = l_rec() / self.diffusion.schedule.max_t
        align_loss = l_align(eps) / self.diffusion.schedule.max_t
        diffuser_loss = self.w * l_diffusion(eps)
        loss = {
            'loss': rec_loss + align_loss + diffuser_loss,
            'rec_loss': rec_loss,
            'align_loss': align_loss,
            'diffuser_loss': diffuser_loss,
        }
        return (loss['loss'], loss)
