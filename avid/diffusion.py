"""Standard U-Net Diffusion. Adapted to 3D."""

# https://github.com/HMUNACHI/nanodl/blob/main/nanodl/__src/models/diffusion.py

import time
from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp


class SinusoidalEmbedding(nn.Module):
    """
    Implements sinusoidal embeddings as a layer in a neural network using JAX.

    This layer generates sinusoidal embeddings based on input positions and a range of frequencies, producing embeddings that capture positional information in a continuous manner. It's particularly useful in models where the notion of position is crucial, such as in generative models for images and audio.

    Attributes:
        embedding_dims (int): The dimensionality of the output embeddings.
        embedding_min_frequency (float): The minimum frequency used in the sinusoidal embedding.
        embedding_max_frequency (float): The maximum frequency used in the sinusoidal embedding.

    Methods:
        setup(): Initializes the layer by computing the angular speeds for the sinusoidal functions based on the specified frequency range.
        __call__(x: jnp.ndarray): Generates the sinusoidal embeddings for the input positions.
    """

    embedding_dims: int
    embedding_min_frequency: float
    embedding_max_frequency: float

    def setup(self):
        num = self.embedding_dims // 2
        start = jnp.log(self.embedding_min_frequency)
        stop = jnp.log(self.embedding_max_frequency)
        frequencies = jnp.exp(jnp.linspace(start, stop, num))
        self.angular_speeds = 2.0 * jnp.pi * frequencies

    def __call__(self, x):
        embeddings = jnp.concatenate(
            [jnp.sin(self.angular_speeds * x), jnp.cos(self.angular_speeds * x)], axis=-1
        )
        return embeddings


class DiffusionBackbone(nn.Module):
    """Abstract model class. Returns denoised images given noisy inputs and noise scale."""

    pass


class DiffusionModel(nn.Module):
    """
    Implements a diffusion model for image generation using JAX.

    Methods:
        setup(): Initializes the diffusion model including the U-Net architecture.
        diffusion_schedule(diffusion_times: jnp.ndarray): Computes the noise and signal rates for given diffusion times.
        denoise(noisy_images: jnp.ndarray, noise_rates: jnp.ndarray, signal_rates: jnp.ndarray): Denoises images given their noise and signal rates.
        __call__(images: jnp.ndarray): Applies the diffusion process to a batch of images.
        reverse_diffusion(initial_noise: jnp.ndarray, diffusion_steps: int): Reverses the diffusion process to generate images from noise.
        generate(num_images: int, diffusion_steps: int): Generates images by reversing the diffusion process from random noise.
    """

    model: DiffusionBackbone
    min_signal_rate: float = 0.02
    max_signal_rate: float = 0.95

    def setup(self):
        self.backbone = self.model.copy()

    def diffusion_schedule(self, diffusion_times: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        start_angle = jnp.arccos(self.max_signal_rate)
        end_angle = jnp.arccos(self.min_signal_rate)
        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)
        signal_rates = jnp.cos(diffusion_angles)
        noise_rates = jnp.sin(diffusion_angles)
        return noise_rates.astype(jnp.bfloat16), signal_rates.astype(jnp.bfloat16)

    def denoise(
        self,
        noisy_images: jnp.ndarray,
        noise_rates: jnp.ndarray,
        signal_rates: jnp.ndarray,
        training: bool,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        pred_noises = self.backbone(noisy_images, noise_rates**2, training=training)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images

    def __call__(self, images: jnp.ndarray, training: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
        key = jax.random.PRNGKey(int(time.time()))
        noises = jax.random.normal(key, shape=images.shape, dtype=jnp.bfloat16)
        batch_size = images.shape[0]
        diffusion_times = jax.random.uniform(key, shape=(batch_size, 1, 1), minval=0.0, maxval=1.0)
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises
        # debug_structure(
        #     n_im=noisy_images, sr=signal_rates, nr=noise_rates, noises=noises, ims=images
        # )
        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, training=training
        )
        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise: jnp.ndarray, diffusion_steps: int) -> jnp.ndarray:
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps
        next_noisy_images = initial_noise

        for step in range(diffusion_steps):
            diffusion_times = jnp.ones((num_images, 1, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(next_noisy_images, noise_rates, signal_rates)
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(next_diffusion_times)
            next_noisy_images = next_signal_rates * pred_images + next_noise_rates * pred_noises

        return pred_images

    def generate(self, images_shape=(1, 24, 24, 24, 64), diffusion_steps: int = 20) -> jnp.ndarray:
        key = jax.random.PRNGKey(int(time.time()))
        noises = jax.random.normal(key, shape=images_shape)

        return self.reverse_diffusion(noises, diffusion_steps)
