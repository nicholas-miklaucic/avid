"""Encoder/decoder for crystal data."""

from typing import Optional
import equinox as eqx
from jaxtyping import Float, Array, jaxtyped, PRNGKeyArray
from beartype import beartype as typechecker
from flax import linen as nn

downsample_factors: list[int] = [2, 2]


class Downsampler(eqx.Module):
    # The input size.
    dim_in: int
    # The input channels.
    chan_in: int
    # The output size.
    dim_out: int
    # The output channels.
    chan_out: int
    # The convolution layer.
    conv: eqx.nn.Conv3d

    def __init__(
        self,
        dim_in: int,
        chan_in: int,
        downsample_factor: int = 2,
        kernel_size: int = 3,
        channel_factor: float = 2,
        *,
        key: PRNGKeyArray,
    ):
        if kernel_size < downsample_factor:
            raise ValueError(
                f'Kernel size {kernel_size} with downsample {downsample_factor} will skip inputs!'
            )
        self.dim_in = dim_in
        self.dim_out = round(self.dim_in / downsample_factor)
        self.chan_in = chan_in
        self.chan_out = round(self.chan_in * channel_factor)
        self.padding =
        self.conv = eqx.nn.Conv3d(
            self.chan_in,
            self.chan_out,
            kernel_size=downsample_factor,
            stride=downsample_factor,
            padding=0,
            key=key,
        )

    def __call__(
        self,
        input: Float[Array, '{self.dim_in} {self.dim_in} {self.dim_in} {self.chan_in}'],
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Float[Array, '{self.dim_out} {self.dim_out} {self.dim_out} {self.chan_in}']:
        return self.conv(input)


class Downsampler(eqx.Module):
    # The input size.
    dim_in: int
    # The input channels.
    chan_in: int
    # The output size.
    dim_out: int
    # The output channels.
    chan_out: int
    # The convolution layer.
    conv: eqx.nn.Conv3d

    def __init__(
        self,
        dim_in: int,
        chan_in: int,
        downsample_factor: int = 2,
        channel_factor: float = 2,
        *,
        key: PRNGKeyArray,
    ):
        self.dim_in = dim_in
        self.dim_out = round(self.dim_in / downsample_factor)
        self.chan_in = chan_in
        self.chan_out = round(self.chan_in * channel_factor)
        self.conv = eqx.nn.Conv3d(
            self.chan_in,
            self.chan_out,
            kernel_size=downsample_factor,
            stride=downsample_factor,
            padding=0,
        )

    def __call__(
        self, input: Float[Array, '{self.dim_in} {self.dim_in} {self.dim_in} {self.chan_in}']
    ) -> Float[Array, '{self.dim_out} {self.dim_out} {self.dim_out} {self.chan_in}']:
        return self.conv(input)
