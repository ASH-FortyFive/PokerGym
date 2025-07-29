from typing import Any, Sequence, SupportsFloat
import gymnasium as gym
import numpy as np
from numpy.typing import NDArray
from gymnasium.spaces import Box

class MaskableBox(Box):
    r"""A (possibly unbounded) box in :math:`\mathbb{R}^n`.

    Extends the standard Box space to allow for masking of actions.
    Mask is another array providing upper and lower bounds for each dimension.
    The mask is applied to the Box space, allowing for actions to be masked out.

    Specifically, a Box represents the Cartesian product of n closed intervals.
    Each interval has the form of one of :math:`[a, b]`, :math:`(-\infty, b]`,
    :math:`[a, \infty)`, or :math:`(-\infty, \infty)`.

    There are two common use cases:

    * Identical bound for each dimension::

        >>> Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        Box(-1.0, 2.0, (3, 4), float32)

    * Independent bound for each dimension::

        >>> Box(low=np.array([-1.0, -2.0]), high=np.array([2.0, 4.0]), dtype=np.float32)
        Box([-1. -2.], [2. 4.], (2,), float32)
    """

    def __init__(
        self,
        low: SupportsFloat | NDArray[Any],
        high: SupportsFloat | NDArray[Any],
        shape: Sequence[int] | None = None,
        dtype: type[np.floating[Any]] | type[np.integer[Any]] = np.float32,
        seed: int | np.random.Generator | None = None,
    ):
        super().__init__(low=low, high=high, shape=shape, dtype=dtype, seed=seed)



    def sample(self, mask: None = None, probability: None = None) -> NDArray[Any]:
        r"""Generates a single random sample inside the Box.

        In creating a sample of the box, each coordinate is sampled (independently) from a distribution
        that is chosen according to the form of the interval:

        * :math:`[a, b]` : uniform distribution
        * :math:`[a, \infty)` : shifted exponential distribution
        * :math:`(-\infty, b]` : shifted negative exponential distribution
        * :math:`(-\infty, \infty)` : normal distribution

        Args:
            mask: A mask for sampling values from the Box space,
            probability: A probability mask for sampling values from the Box space, currently unsupported.

        Returns:
            A sampled value from the Box
        """
        if probability is not None:
            raise gym.error.Error(
                f"Box.sample cannot be provided a probability mask, actual value: {probability}"
            )

        high = self.high if self.dtype.kind == "f" else self.high.astype("int64") + 1
        low = self.low if self.dtype.kind == "f" else self.low.astype("int64")

        if mask is not None:
            assert mask.shape == (2,), "Only 2D masks are supported."
            mask_low, mask_high = mask
            low = np.maximum(low, np.asarray(mask_low, dtype=self.dtype))
            high = np.minimum(high, np.asarray(mask_high, dtype=self.dtype))
            # Validate that low <= high
            if not np.all(low <= high):
                raise ValueError("Mask lower bounds must be less than or equal to upper bounds.")

        sample = np.empty(self.shape)

        # Masking arrays which classify the coordinates according to interval type
        unbounded = ~self.bounded_below & ~self.bounded_above
        upp_bounded = ~self.bounded_below & self.bounded_above
        low_bounded = self.bounded_below & ~self.bounded_above
        bounded = self.bounded_below & self.bounded_above

        # Vectorized sampling by interval type
        sample[unbounded] = self.np_random.normal(size=unbounded[unbounded].shape)

        sample[low_bounded] = (
            self.np_random.exponential(size=low_bounded[low_bounded].shape)
            + low[low_bounded]
        )

        sample[upp_bounded] = (
            -self.np_random.exponential(size=upp_bounded[upp_bounded].shape)
            + high[upp_bounded]
        )

        sample[bounded] = self.np_random.uniform(
            low=low[bounded], high=high[bounded], size=bounded[bounded].shape
        )

        if self.dtype.kind in ["i", "u", "b"]:
            sample = np.floor(sample)

        # clip values that would underflow/overflow
        if np.issubdtype(self.dtype, np.signedinteger):
            dtype_min = np.iinfo(self.dtype).min + 2
            dtype_max = np.iinfo(self.dtype).max - 2
            sample = sample.clip(min=dtype_min, max=dtype_max)
        elif np.issubdtype(self.dtype, np.unsignedinteger):
            dtype_min = np.iinfo(self.dtype).min
            dtype_max = np.iinfo(self.dtype).max
            sample = sample.clip(min=dtype_min, max=dtype_max)

        sample = sample.astype(self.dtype)

        # float64 values have lower than integer precision near int64 min/max, so clip
        # again in case something has been cast to an out-of-bounds value
        if self.dtype == np.int64:
            sample = sample.clip(min=self.low, max=self.high)
        return sample
