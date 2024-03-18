"""Diffusion layers and modules."""

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.struct import dataclass

from avid.layers import LazyInMLP
from avid.vit import Encoder, ImageEmbed


@dataclass
class DiffusionInput:
    """Input to a diffusion step."""

    # Noised latent.
    x_t: jax.Array
    # Time step, as a float.
    t: jax.Array
    # Class/label, as an int.
    y: jax.Array


# https://github.com/facebookresearch/DiT/blob/main/models.py
class TimestepEmbed(nn.Module):
    frequency_embed_dim: int
    mlp = LazyInMLP

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = jnp.exp(
            -jnp.log(max_period) * jnp.arange(start=0, end=half, dtype=jnp.float32) / half
        )
        args = t[:, None] * freqs[None]
        embedding = jnp.concat([jnp.cos(args), jnp.sin(args)], dim=-1)
        if dim % 2:
            embedding = jnp.concat([embedding, jnp.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    @nn.compact
    def __call__(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embed_dim)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbed(nn.Module):
    """Diffusion class label embedding: reserves an embedding for a Ø class."""

    # Number of actual classes: one is added for the null class.
    num_classes: int

    # Embed dimension.
    embed_dim: int

    @nn.compact
    def __call__(self, y: jax.Array) -> jax.Array:
        self.embeddings = nn.Embed(self.num_classes + 1, self.embed_dim)
        return self.embeddings(y)


class DiT(nn.Module):
    im_embed: ImageEmbed
    encoder: Encoder
    time_embed: TimestepEmbed
    label_embed: LabelEmbed
    modulation_projection: LazyInMLP
    out_projection: LazyInMLP

    @nn.compact
    def __call__(self, data: DiffusionInput, training: bool):
        t = self.time_embed(data.t)
        y = self.label_embed(data.y)
        c = t + y
        hidden = self.vit.im_embed.inner.patch_latent_dim
        self.modulation_projection.out_dim = hidden
        aby = self.modulation_projection(c, training)

        latent = self.im_embed(data.x_t)
