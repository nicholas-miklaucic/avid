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


class UNetResidualBlock(nn.Module):
    """
    Implements a residual block within a U-Net architecture using JAX.

    This module defines a residual block with convolutional layers and normalization, followed by a residual connection. It's a fundamental building block in constructing deeper and more complex U-Net architectures for tasks like image segmentation and generation.

    Attributes:
        width (int): The number of output channels for the convolutional layers within the block.

    Methods:
        __call__(x: jnp.ndarray): Processes the input tensor through the residual block and returns the result.
    """

    width: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        input_width = x.shape[-1]

        # Define layers
        convolution_1 = nn.Conv(self.width, kernel_size=(1, 1, 1))
        convolution_2 = nn.Conv(self.width, kernel_size=(3, 3, 3), padding='CIRCULAR')
        convolution_3 = nn.Conv(self.width, kernel_size=(3, 3, 3), padding='CIRCULAR')
        norm = nn.GroupNorm(num_groups=2, epsilon=1e-5, use_bias=False, use_scale=False)

        # Residual connection
        residual = convolution_1(x) if input_width != self.width else x

        # Forward pass
        x = norm(x)
        x = nn.swish(x)
        x = convolution_2(x)
        x = nn.swish(x)
        x = convolution_3(x)

        return x + residual


class UNetDownBlock(nn.Module):
    """
    Implements a down-sampling block in a U-Net architecture using JAX.

    This module consists of a sequence of residual blocks followed by an average pooling operation to reduce the spatial dimensions. It's used to capture higher-level features at reduced spatial resolutions in the encoding pathway of a U-Net.

    Attributes:
        width (int): The number of output channels for the convolutional layers within the block.
        block_depth (int): The number of residual blocks to include in the down-sampling block.

    Methods:
        setup(): Initializes the sequence of residual blocks.
        __call__(x: jnp.ndarray): Processes the input tensor through the down-sampling block and returns the result.
    """

    width: int
    block_depth: int

    def setup(self):
        self.residual_blocks = [UNetResidualBlock(self.width) for _ in range(self.block_depth)]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for block in self.residual_blocks:
            x = block(x)
        x = nn.avg_pool(x, window_shape=(2, 2, 2), strides=(2, 2, 2))
        return x


class UNetUpBlock(nn.Module):
    """
    Implements an up-sampling block in a U-Net architecture using JAX.

    This module consists of a sequence of residual blocks and a bilinear up-sampling operation to increase the spatial dimensions. It's used in the decoding pathway of a U-Net to progressively recover spatial resolution and detail in the output image.

    Attributes:
        width (int): The number of output channels for the convolutional layers within the block.
        block_depth (int): The number of residual blocks to include in the up-sampling block.

    Methods:
        setup(): Initializes the sequence of residual blocks.
        __call__(x: jnp.ndarray, skip: jnp.ndarray): Processes the input tensor and a skip connection from the encoding pathway through the up-sampling block and returns the result.
    """

    width: int
    block_depth: int

    def setup(self):
        self.residual_blocks = [UNetResidualBlock(self.width) for _ in range(self.block_depth)]

    def __call__(self, x: jnp.ndarray, skip: jnp.ndarray) -> jnp.ndarray:
        B, D, H, W, C = x.shape
        upsampled_shape = (B, D * 2, H * 2, W * 2, C)
        x = jax.image.resize(x, shape=upsampled_shape, method='bilinear')
        x = jnp.concatenate([x, skip], axis=-1)
        for block in self.residual_blocks:
            x = block(x)
        return x


class UNet(nn.Module):
    """
    Implements the U-Net architecture for image processing tasks using JAX.

    This model is widely used for tasks such as image segmentation, denoising, and super-resolution. It features a symmetric encoder-decoder structure with skip connections between corresponding layers in the encoder and decoder to preserve spatial information.

    Attributes:
        image_size (Tuple[int, int]): The size of the input images (height, width).
        widths (List[int]): The number of output channels for each block in the U-Net architecture.
        block_depth (int): The number of residual blocks in each down-sampling and up-sampling block.
        embed_dims (int): The dimensionality of the sinusoidal embeddings for encoding positional information.
        embed_min_freq (float): The minimum frequency for the sinusoidal embeddings.
        embed_max_freq (float): The maximum frequency for the sinusoidal embeddings.

    Methods:
        setup(): Initializes the U-Net architecture including the sinusoidal embedding layer, down-sampling blocks, residual blocks, and up-sampling blocks.
        __call__(noisy_images: jnp.ndarray, noise_variances: jnp.ndarray): Processes noisy images and their associated noise variances through the U-Net and returns the denoised images.
    """

    widths: tuple[int]
    block_depth: int
    embed_dims: int
    embed_min_freq: float
    embed_max_freq: float

    @nn.compact
    def __call__(self, noisy_images: jnp.ndarray, noise_variances: jnp.ndarray) -> jnp.ndarray:
        sinusoidal_embedding = SinusoidalEmbedding(
            self.embed_dims, self.embed_min_freq, self.embed_max_freq
        )
        down_blocks = [UNetDownBlock(width, self.block_depth) for width in self.widths[:-1]]
        residual_blocks = [UNetResidualBlock(self.widths[-1]) for _ in range(self.block_depth)]
        up_blocks = [UNetUpBlock(width, self.block_depth) for width in reversed(self.widths[:-1])]
        convolution_1 = nn.Conv(self.widths[0], kernel_size=(1, 1, 1))
        convolution_2 = nn.Conv(
            noisy_images.shape[-1], kernel_size=(1, 1, 1), kernel_init=nn.initializers.zeros
        )
        e = sinusoidal_embedding(noise_variances)
        upsampled_shape = (
            # B, D, H, W
            *noisy_images.shape[:4],
            self.embed_dims,
        )
        e = jax.image.resize(e, upsampled_shape, method='nearest')

        x = convolution_1(noisy_images)
        x = jnp.concatenate([x, e], axis=-1)

        skips = []
        for block in down_blocks:
            skips.append(x)
            x = block(x)

        for block in residual_blocks:
            x = block(x)

        for block, skip in zip(up_blocks, reversed(skips)):
            x = block(x, skip)

        outputs = convolution_2(x)
        return outputs


class DiffusionModel(nn.Module):
    """
    Implements a diffusion model for image generation using JAX.

    Diffusion models are a class of generative models that learn to denoise images through a gradual process of adding and removing noise. This implementation uses a U-Net architecture for the denoising process and supports custom diffusion schedules.

    Attributes:
        image_size (int): The size of the generated images.
        widths (List[int]): The number of output channels for each block in the U-Net architecture.
        block_depth (int): The number of residual blocks in each down-sampling and up-sampling block.
        min_signal_rate (float): The minimum signal rate in the diffusion process.
        max_signal_rate (float): The maximum signal rate in the diffusion process.
        embed_dims (int): The dimensionality of the sinusoidal embeddings for encoding noise levels.
        embed_min_freq (float): The minimum frequency for the sinusoidal embeddings.
        embed_max_freq (float): The maximum frequency for the sinusoidal embeddings.

    Methods:
        setup(): Initializes the diffusion model including the U-Net architecture.
        diffusion_schedule(diffusion_times: jnp.ndarray): Computes the noise and signal rates for given diffusion times.
        denoise(noisy_images: jnp.ndarray, noise_rates: jnp.ndarray, signal_rates: jnp.ndarray): Denoises images given their noise and signal rates.
        __call__(images: jnp.ndarray): Applies the diffusion process to a batch of images.
        reverse_diffusion(initial_noise: jnp.ndarray, diffusion_steps: int): Reverses the diffusion process to generate images from noise.
        generate(num_images: int, diffusion_steps: int): Generates images by reversing the diffusion process from random noise.

    Example usage:
        ```
        import jax
        import jax.numpy as jnp
        from nanodl import ArrayDataset, DataLoader
        from nanodl import DiffusionModel, DiffusionDataParallelTrainer

        image_size = 32
        block_depth = 2
        batch_size = 8
        widths = [32, 64, 128]
        key = jax.random.PRNGKey(0)
        input_shape = (101, image_size, image_size, 3)
        images = jax.random.normal(key, input_shape)

        # Use your own images
        dataset = ArrayDataset(images)
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                drop_last=False)

        # Create diffusion model
        diffusion_model = DiffusionModel(image_size, widths, block_depth)
        params = diffusion_model.init(key, images)
        pred_noises, pred_images = diffusion_model.apply(params, images)
        print(pred_noises.shape, pred_images.shape)

        # Training on your data
        # Note: saved params are often different from training weights, use the saved params for generation
        trainer = DiffusionDataParallelTrainer(diffusion_model,
                                            input_shape=images.shape,
                                            weights_filename='params.pkl',
                                            learning_rate=1e-4)
        trainer.train(dataloader, 10, dataloader)
        print(trainer.evaluate(dataloader))

        # Generate some samples
        params = trainer.load_params('params.pkl')
        generated_images = diffusion_model.apply({'params': params},
                                                num_images=5,
                                                diffusion_steps=5,
                                                method=diffusion_model.generate)
        print(generated_images.shape)
        ```
    """

    model: UNet
    min_signal_rate: float = 0.02
    max_signal_rate: float = 0.95

    def setup(self):
        self.unet = self.model.copy()

    def diffusion_schedule(self, diffusion_times: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        start_angle = jnp.arccos(self.max_signal_rate)
        end_angle = jnp.arccos(self.min_signal_rate)
        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)
        signal_rates = jnp.cos(diffusion_angles)
        noise_rates = jnp.sin(diffusion_angles)
        return noise_rates, signal_rates

    def denoise(
        self, noisy_images: jnp.ndarray, noise_rates: jnp.ndarray, signal_rates: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        pred_noises = self.unet(noisy_images, noise_rates**2)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images

    def __call__(self, images: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        key = jax.random.PRNGKey(int(time.time()))
        noises = jax.random.normal(key, shape=images.shape)
        batch_size = images.shape[0]
        diffusion_times = jax.random.uniform(
            key, shape=(batch_size, 1, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises
        pred_noises, pred_images = self.denoise(noisy_images, noise_rates, signal_rates)
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
