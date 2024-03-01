"""Vision transformer modules. https://docs.kidger.site/equinox/examples/vision_transformer/"""

import functools

import einops  # https://github.com/arogozhnikov/einops
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pyrallis
from avid.config import MainConfig
from avid.dataset import dataloader, num_elements_class
import optax  # https://github.com/deepmind/optax
import plotext as plt

from jaxtyping import Array, Float, PRNGKeyArray


config = pyrallis.parse(MainConfig)
config.cli.set_up_logging()

# Hyperparameters
lr = 0.0001
dropout_rate = 0.1
beta1 = 0.9
beta2 = 0.999
batch_size = config.batch_size
patch_size = 6
num_patches = 64
num_steps = 3_000
image_size = tuple([config.data_encoder.n_grid] * 3)
embedding_dim = 512
hidden_dim = 256
num_heads = 8
num_layers = 6
height, width, channels = image_size
num_classes = 4


class PatchEmbedding(eqx.Module):
    linear: eqx.nn.Embedding
    patch_size: int

    def __init__(
        self,
        input_channels: int,
        output_shape: int,
        patch_size: int,
        key: PRNGKeyArray,
    ):
        self.patch_size = patch_size

        self.linear = eqx.nn.Linear(
            self.patch_size**2 * input_channels,
            output_shape,
            key=key,
        )

    def __call__(
        self, x: Float[Array, 'channels height width']
    ) -> Float[Array, 'num_patches embedding_dim']:
        x = einops.rearrange(
            x,
            'c (h ph) (w pw) -> (h w) (c ph pw)',
            ph=self.patch_size,
            pw=self.patch_size,
        )
        x = jax.vmap(self.linear)(x)

        return x


class AttentionBlock(eqx.Module):
    layer_norm1: eqx.nn.LayerNorm
    layer_norm2: eqx.nn.LayerNorm
    attention: eqx.nn.MultiheadAttention
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    dropout1: eqx.nn.Dropout
    dropout2: eqx.nn.Dropout

    def __init__(
        self,
        input_shape: int,
        hidden_dim: int,
        num_heads: int,
        dropout_rate: float,
        key: PRNGKeyArray,
    ):
        key1, key2, key3 = jr.split(key, 3)

        self.layer_norm1 = eqx.nn.LayerNorm(input_shape)
        self.layer_norm2 = eqx.nn.LayerNorm(input_shape)
        self.attention = eqx.nn.MultiheadAttention(num_heads, input_shape, key=key1)

        self.linear1 = eqx.nn.Linear(input_shape, hidden_dim, key=key2)
        self.linear2 = eqx.nn.Linear(hidden_dim, input_shape, key=key3)
        self.dropout1 = eqx.nn.Dropout(dropout_rate)
        self.dropout2 = eqx.nn.Dropout(dropout_rate)

    def __call__(
        self,
        x: Float[Array, 'num_patches embedding_dim'],
        enable_dropout: bool,
        key: PRNGKeyArray,
    ) -> Float[Array, 'num_patches embedding_dim']:
        input_x = jax.vmap(self.layer_norm1)(x)
        x = x + self.attention(input_x, input_x, input_x)

        input_x = jax.vmap(self.layer_norm2)(x)
        input_x = jax.vmap(self.linear1)(input_x)
        input_x = jax.nn.gelu(input_x)

        key1, key2 = jr.split(key, num=2)

        input_x = self.dropout1(input_x, inference=not enable_dropout, key=key1)
        input_x = jax.vmap(self.linear2)(input_x)
        input_x = self.dropout2(input_x, inference=not enable_dropout, key=key2)

        x = x + input_x

        return x


class VisionTransformer(eqx.Module):
    patch_embedding: PatchEmbedding
    positional_embedding: jnp.ndarray
    cls_token: jnp.ndarray
    attention_blocks: list[AttentionBlock]
    dropout: eqx.nn.Dropout
    mlp: eqx.nn.Sequential
    num_layers: int

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout_rate: float,
        patch_size: int,
        num_patches: int,
        num_classes: int,
        key: PRNGKeyArray,
    ):
        key1, key2, key3, key4, key5 = jr.split(key, 5)

        self.patch_embedding = PatchEmbedding(channels, embedding_dim, patch_size, key1)

        self.positional_embedding = jr.normal(key2, (num_patches + 1, embedding_dim))

        self.cls_token = jr.normal(key3, (1, embedding_dim))

        self.num_layers = num_layers

        self.attention_blocks = [
            AttentionBlock(embedding_dim, hidden_dim, num_heads, dropout_rate, key4)
            for _ in range(self.num_layers)
        ]

        self.dropout = eqx.nn.Dropout(dropout_rate)

        self.mlp = eqx.nn.Sequential(
            [
                eqx.nn.LayerNorm(embedding_dim),
                eqx.nn.Linear(embedding_dim, num_classes, key=key5),
            ]
        )

    def __call__(
        self,
        x: Float[Array, 'channels height width'],
        enable_dropout: bool,
        key: PRNGKeyArray,
    ) -> Float[Array, 'num_classes']:
        x = self.patch_embedding(x)

        x = jnp.concatenate((self.cls_token, x), axis=0)

        x += self.positional_embedding[
            : x.shape[0]
        ]  # Slice to the same length as x, as the positional embedding may be longer.

        dropout_key, *attention_keys = jr.split(key, num=self.num_layers + 1)

        x = self.dropout(x, inference=not enable_dropout, key=dropout_key)

        for block, attention_key in zip(self.attention_blocks, attention_keys):
            x = block(x, enable_dropout, key=attention_key)

        x = x[0]  # Select the CLS token.
        x = self.mlp(x)

        return x


@eqx.filter_value_and_grad
def compute_grads(model: VisionTransformer, images: jnp.ndarray, labels: jnp.ndarray, key):
    logits = jax.vmap(model, in_axes=(0, None, 0))(images, True, key)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)

    return jnp.mean(loss)


@eqx.filter_jit
def step_model(
    model: VisionTransformer,
    optimizer: optax.GradientTransformation,
    state: optax.OptState,
    images: jnp.ndarray,
    labels: jnp.ndarray,
    key,
):
    loss, grads = compute_grads(model, images, labels, key)
    updates, new_state = optimizer.update(grads, state, model)

    model = eqx.apply_updates(model, updates)

    return model, new_state, loss


def train(
    model: VisionTransformer,
    optimizer: optax.GradientTransformation,
    state: optax.OptState,
    data_loader,
    num_steps: int,
    key=None,
):
    losses = []

    for step, batch in zip(range(num_steps), data_loader):
        key, *subkeys = jr.split(key, num=config.batch_size + 1)
        subkeys = jnp.array(subkeys)

        dens_img = einops.rearrange(
            batch['density'], 'batch (h w c) -> batch h w c', h=height, w=width, c=channels
        )

        (model, state, loss) = step_model(
            model, optimizer, state, dens_img, batch['n_elements_label'], subkeys
        )

        losses.append(loss.item())

        print_every = num_steps // 20
        if (step + 1) % print_every == 0:
            plt.plot(np.arange(step), losses)
            plt.title('Loss')
            plt.show()
            plt.clear_figure()

    return model, state, losses


key = jr.PRNGKey(2003)

model = VisionTransformer(
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    dropout_rate=dropout_rate,
    patch_size=patch_size,
    num_patches=num_patches,
    num_classes=num_classes,
    key=key,
)

optimizer = optax.adamw(
    learning_rate=lr,
    b1=beta1,
    b2=beta2,
)


state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

if __name__ == '__main__':
    from rich.prompt import Confirm

    if Confirm.ask('Train from scratch?'):
        model, state, losses = train(
            model, optimizer, state, dataloader(config), num_steps, key=key
        )

        eqx.tree_serialise_leaves('models/vit_num_elems1.eqx', model)
    else:
        model = eqx.tree_deserialise_leaves('models/vit_num_elems1.eqx', model)

    accuracies = []
    for batch in dataloader(config, split='test'):
        dens_img = einops.rearrange(
            batch['density'], 'batch (h w c) -> batch h w c', h=height, w=width, c=channels
        )
        logits = jax.vmap(model, in_axes=(0, None, None))(dens_img, False, key)
        predictions = jnp.argmax(logits, axis=-1)
        accuracy = jnp.mean(predictions == batch['n_elements_label'])
        accuracies.append(accuracy)

    print(f'Accuracy: {np.sum(accuracies) / len(accuracies) * 100}%')
