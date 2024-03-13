# AVID

## Acronym Very Incomplete: Diffusion?

Work on applying diffusion and vision transformers to materials. Right now, I'm
just doing cubic perovskites. Cool stuff TBD.

## Installation

[Install JAX](https://jax.readthedocs.io/en/latest/installation.html), and then
install the other necessary libraries:

```bash
pip install toml pyrallis rho-plus equinox diffrax optax plotext rich einops pandas plotly matplotlib seaborn pymatgen tfp-nightly[jax] textual textual-dev textual-plotext flax orbax beartype humanize clu pyarrow
```

TODO:

- [ ] Move things into config files
- [ ] Take a look at latent space, both species and encoder
- [ ] Implement upsampler and diffusion transformer backend
- [ ] Implement warmstart/encoder warmstart
- [ ] Perhaps configure DiLED in addition to normal diffusion and encoder losses
- [ ] EMA/SWA
