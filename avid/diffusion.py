"""Diffusion training loop. https://docs.kidger.site/equinox/examples/score_based_diffusion/"""

import array
import functools as ft
import gzip
import os
import struct
import urllib.request

import numpy as np
import pyrallis

from avid.config import MainConfig
from avid.dataset import dataloader
from avid.unet import UNet
import diffrax as dfx  # https://github.com/patrick-kidger/diffrax
import einops  # https://github.com/arogozhnikov/einops
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import plotext as plt
import optax  # https://github.com/deepmind/optax


def single_loss_fn(model, weight, int_beta, data, t, key):
    mean = data * jnp.exp(-0.5 * int_beta(t))
    var = jnp.maximum(1 - jnp.exp(-int_beta(t)), 1e-5)
    std = jnp.sqrt(var)
    noise = jr.normal(key, data.shape)
    y = mean + std * noise
    pred = model(t, y, key=key)
    return weight(t) * jnp.mean((pred + noise / std) ** 2)


def batch_loss_fn(model, weight, int_beta, data, t1, key):
    batch_size = data.shape[0]
    tkey, losskey = jr.split(key)
    losskey = jr.split(losskey, batch_size)
    # Low-discrepancy sampling over t to reduce variance
    t = jr.uniform(tkey, (batch_size,), minval=0, maxval=t1 / batch_size)
    t = t + (t1 / batch_size) * jnp.arange(batch_size)
    loss_fn = ft.partial(single_loss_fn, model, weight, int_beta)
    loss_fn = jax.vmap(loss_fn)
    return jnp.mean(loss_fn(data, t, losskey))


@eqx.filter_jit
def single_sample_fn(model, int_beta, data_shape, dt0, t1, key):
    def drift(t, y, args):
        _, beta = jax.jvp(int_beta, (t,), (jnp.ones_like(t),))
        return -0.5 * beta * (y + model(t, y))

    term = dfx.ODETerm(drift)
    solver = dfx.Tsit5()
    t0 = 0
    y1 = jr.normal(key, data_shape)
    # reverse time, solve from t1 to t0
    sol = dfx.diffeqsolve(term, solver, t1, t0, -dt0, y1)
    return sol.ys[0]


@eqx.filter_jit
def make_step(model, weight, int_beta, data, t1, key, opt_state, opt_update):
    loss_fn = eqx.filter_value_and_grad(batch_loss_fn)
    loss, grads = loss_fn(model, weight, int_beta, data, t1, key)
    updates, opt_state = opt_update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    key = jr.split(key, 1)[0]
    return loss, model, key, opt_state


def main(
    config: MainConfig,
    # Model hyperparameters
    t1=10.0,
    # Optimisation hyperparameters
    num_steps=300,
    lr=3e-4,
    # Sampling hyperparameters
    # Seed
    seed=5678,
):
    key = jr.PRNGKey(seed)
    model_key, train_key, loader_key, sample_key = jr.split(key, 4)

    h = w = d = config.voxelizer.n_grid

    int_beta = lambda t: t  # Try experimenting with other options here!
    weight = lambda t: 1 - jnp.exp(-int_beta(t))  # Just chosen to upweight the region near t=0.

    model = UNet(
        data_shape=(1, h, w, d),
        is_biggan=False,
        dim_mults=[1, 1],
        hidden_size=4,
        heads=4,
        dim_head=4,
        dropout_rate=0.1,
        num_res_blocks=2,
        attn_resolutions={3},
        key=model_key,
    )

    print(model)

    opt = optax.adabelief(lr)
    # Optax will update the floating-point JAX arrays in the model.
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    losses = []

    for step, batch in zip(range(num_steps), dataloader(config)):
        key, *subkeys = jr.split(key, num=config.batch_size + 1)
        subkeys = jnp.array(subkeys)

        dens_img = einops.rearrange(
            batch['density'], 'batch (h w d) -> batch 1 h w d', h=h, w=w, d=d
        )

        value, model, train_key, opt_state = make_step(
            model, weight, int_beta, dens_img, t1, train_key, opt_state, opt.update
        )

        losses.append(value.item())

        print_every = num_steps // 20
        if (step + 1) % print_every == 0:
            plt.plot(np.arange(step), losses)
            plt.title('Loss')
            plt.show()
            plt.clear_figure()

    eqx.tree_serialise_leaves('models/diffusion1.eqx', model)
    return model


def sample(model, config, seed, sample_size: int = 10, dt0=0.1, t1=10.0):
    key = jr.PRNGKey(seed)
    int_beta = lambda t: t  # Try experimenting with other options here!
    h = w = d = config.data_encoder.n_grid
    sample_key = jr.split(key, sample_size**3)
    sample_fn = ft.partial(single_sample_fn, model, int_beta, (1, h, w, d), dt0, t1)
    sample = jax.vmap(sample_fn)(sample_key)
    sample = jnp.clip(sample, 0.0, 1.0)
    sample = einops.rearrange(
        sample,
        '(n1 n2 n3) 1 h w d -> (n1 h) (n2 w) (n3 d)',
        n1=sample_size,
        n2=sample_size,
        n3=sample_size,
    )


if __name__ == '__main__':
    from rich.prompt import Confirm

    config = pyrallis.parse(MainConfig)

    if Confirm.ask('Run diffusion?'):
        main(config)
    else:
        h = w = d = config.voxelizer.n_grid
        model = UNet(
            data_shape=(1, h, w, d),
            is_biggan=False,
            dim_mults=[1, 1],
            hidden_size=4,
            heads=4,
            dim_head=4,
            dropout_rate=0.1,
            num_res_blocks=2,
            attn_resolutions={3},
            key=jr.PRNGKey(2345),
        )

        model = eqx.tree_deserialise_leaves('models/diffusion1.eqx', model)
