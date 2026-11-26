"""Per-unit activation quantiles over a corpus, and a hook that ablates a quantile band.

Each unit at a hook (e.g. each MLP neuron at ``blocks.N.mlp.hook_post``) gets thresholds from its
own activation distribution. :class:`ActivationQuantiles` estimates them in bounded memory over
any number of batches; :func:`band_ablation_hook` then replaces every activation inside its unit's
central band (say between the unit's 3rd and 97th percentiles) and keeps only the tails, on
either sign.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

import torch
from jaxtyping import Float

from transformer_lens.hook_points import HookPoint

Threshold = Union[float, torch.Tensor]

_MAX_QUANTILE_INPUT = 2**24


class ActivationQuantiles:
    """Streaming per-unit quantiles of activations whose last dimension indexes units.

    Keeps a uniform random sample (without replacement) of at most ``max_samples`` rows, where a
    row is one unit vector such as one token's ``d_mlp`` neuron activations. While no more than
    ``max_samples`` rows have been seen the sample is every row, so :meth:`quantiles` is exact;
    beyond that it is computed on the sample. :attr:`mean` is always exact over every row seen.
    The sample is ``max_samples × d`` float32 values on the device of the first batch, and
    :meth:`update` and :meth:`quantiles` briefly need about twice that. A NaN activation makes
    its unit's quantiles and mean NaN.

    Examples:
        >>> import torch
        >>> from transformer_lens.utilities import ActivationQuantiles
        >>> stats = ActivationQuantiles(max_samples=1000)
        >>> stats.update(torch.arange(10.0).reshape(5, 2))  # 5 rows, 2 units
        >>> stats.quantiles([0.0, 0.5, 1.0])
        tensor([[0., 1.],
                [4., 5.],
                [8., 9.]])
        >>> stats.mean
        tensor([4., 5.])

    With a model, update from the cache batch by batch, passing the attention mask so padding
    positions are left out::

        stats = ActivationQuantiles(max_samples=50_000)
        for tokens, attention_mask in batches:
            _, cache = model.run_with_cache(tokens, names_filter=hook_name)
            stats.update(cache[hook_name], mask=attention_mask.bool())
        low, high = stats.quantiles([0.03, 0.97])
    """

    def __init__(self, max_samples: int = 100_000, *, generator: Optional[torch.Generator] = None):
        """
        Args:
            max_samples: Maximum number of rows kept for the quantile estimate, at most
                ``2**24`` (the largest input ``torch.quantile`` accepts).
            generator: CPU generator for the sampling keys, for reproducible estimates once more
                than ``max_samples`` rows have been seen. Defaults to torch's global generator,
                which ``update`` then advances.
        """
        if isinstance(max_samples, bool) or not isinstance(max_samples, int):
            raise ValueError(f"max_samples must be a positive integer, got {max_samples!r}")
        if not 1 <= max_samples <= _MAX_QUANTILE_INPUT:
            raise ValueError(
                f"max_samples must be between 1 and {_MAX_QUANTILE_INPUT}, got {max_samples}"
            )
        if generator is not None and generator.device.type != "cpu":
            raise ValueError(f"generator must be a CPU generator, got {generator.device}")
        self.max_samples = max_samples
        self.generator = generator
        self.n_seen = 0
        self._sample: Optional[torch.Tensor] = None
        # Rows are kept by smallest random key (bottom-k sampling): a uniform sample of the stream.
        # Keys and the running sum stay on CPU in float64, which MPS does not support.
        self._keys: Optional[torch.Tensor] = None
        self._sum: Optional[torch.Tensor] = None

    def update(
        self,
        activations: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Add a batch of activations; every dimension but the last indexes rows.

        Args:
            activations: Floating-point activations, e.g. ``[batch, pos, d_mlp]``.
            mask: Optional boolean tensor of shape ``activations.shape[:-1]``; rows where it is
                ``False`` (such as padding positions) are skipped.
        """
        if not isinstance(activations, torch.Tensor) or not activations.is_floating_point():
            raise TypeError("activations must be a floating-point tensor")
        if activations.ndim == 0:
            raise ValueError("activations need a last dimension indexing units")
        rows = activations.detach().reshape(-1, activations.shape[-1])
        if mask is not None:
            if mask.dtype != torch.bool or mask.shape != activations.shape[:-1]:
                raise ValueError(
                    f"mask must be a bool tensor of shape {tuple(activations.shape[:-1])}, "
                    f"got {mask.dtype} of shape {tuple(mask.shape)}"
                )
            rows = rows[mask.reshape(-1).to(rows.device)]
        if self._sample is not None:
            if rows.shape[-1] != self._sample.shape[-1]:
                raise ValueError(
                    f"expected {self._sample.shape[-1]} units per row, got {rows.shape[-1]}"
                )
            rows = rows.to(self._sample.device)
        if rows.shape[0] == 0:
            return
        # Copy, so the sample never aliases a tensor the caller may reuse or edit.
        rows = rows.to(torch.float32, copy=True)

        keys = torch.rand(rows.shape[0], generator=self.generator, dtype=torch.float64)
        batch_sum = rows.sum(0).cpu().double()
        if self._sample is None or self._keys is None or self._sum is None:
            sample, all_keys = rows, keys
            self._sum = batch_sum
        else:
            sample = torch.cat([self._sample, rows])
            all_keys = torch.cat([self._keys, keys])
            self._sum += batch_sum
        if sample.shape[0] > self.max_samples:
            keep = all_keys.topk(self.max_samples, largest=False).indices
            sample, all_keys = sample[keep.to(sample.device)], all_keys[keep]
        self._sample, self._keys = sample, all_keys
        self.n_seen += rows.shape[0]

    @property
    def n_samples(self) -> int:
        """Number of rows currently in the sample (``min(n_seen, max_samples)``)."""
        return 0 if self._sample is None else self._sample.shape[0]

    @property
    def mean(self) -> Float[torch.Tensor, "d"]:
        """Exact per-unit mean over every row seen, e.g. a value for mean-ablation."""
        if self._sum is None or self._sample is None:
            raise ValueError("no activations have been added yet")
        return (self._sum / self.n_seen).to(self._sample.device, torch.float32)

    def quantiles(self, q: Union[float, Sequence[float], torch.Tensor]) -> torch.Tensor:
        """Per-unit quantiles at levels ``q`` in ``[0, 1]``, linearly interpolated.

        Returns:
            Shape ``[d]`` for a scalar ``q``, otherwise ``[len(q), d]``.
        """
        if self._sample is None:
            raise ValueError("no activations have been added yet")
        levels = torch.as_tensor(q, dtype=self._sample.dtype, device=self._sample.device)
        if levels.ndim > 1:
            raise ValueError("q must be a scalar or one-dimensional")
        if not bool(((levels >= 0) & (levels <= 1)).all()):
            raise ValueError(f"quantile levels must be in [0, 1], got {q!r}")
        return torch.quantile(self._sample, levels, dim=0)


def band_ablation_hook(
    low: Threshold,
    high: Threshold,
    value: Threshold = 0.0,
) -> Callable[..., torch.Tensor]:
    """Forward hook that replaces activations inside a per-unit band and keeps the tails.

    An activation ``x`` of unit ``i`` is replaced by ``value`` when ``low[i] <= x <= high[i]``,
    so only the firings below ``low`` or above ``high`` pass through. With ``low`` and ``high``
    at each neuron's 3rd and 97th percentiles (from :class:`ActivationQuantiles`) this keeps each
    neuron's strongest firings of both signs. The input is not modified in place, so a tensor
    already stored in a cache stays intact.

    Args:
        low: Lower edge of the band: a float, or a tensor of shape ``[d]`` (one per unit).
        high: Upper edge, same form as ``low`` and ``>= low`` everywhere.
        value: Replacement inside the band: a float or a ``[d]`` tensor, e.g.
            :attr:`ActivationQuantiles.mean` for mean-ablation. Defaults to ``0.0``.

    Returns:
        A hook function for ``run_with_hooks`` / ``add_hook`` at a hook whose last dimension
        indexes the same units, e.g. ``blocks.N.mlp.hook_post``.

    Examples:
        >>> import torch
        >>> from transformer_lens.utilities import band_ablation_hook
        >>> hook_fn = band_ablation_hook(torch.tensor([-1.0, 0.0]), torch.tensor([1.0, 2.0]))
        >>> hook_fn(torch.tensor([[-3.0, 1.0], [0.5, 5.0]]), hook=None)
        tensor([[-3.,  0.],
                [ 0.,  5.]])
    """
    low_t, high_t, value_t = (torch.as_tensor(t) for t in (low, high, value))
    for name, t in (("low", low_t), ("high", high_t), ("value", value_t)):
        if t.ndim > 1:
            raise ValueError(f"{name} must be a scalar or a [d] tensor, got shape {tuple(t.shape)}")
        if not bool(torch.isfinite(t).all()):
            raise ValueError(f"{name} must be finite")
    sizes = {t.shape[0] for t in (low_t, high_t, value_t) if t.ndim == 1}
    if len(sizes) > 1:
        raise ValueError(f"low, high and value must have the same number of units, got {sizes}")
    if not bool((low_t.cpu() <= high_t.cpu()).all()):
        raise ValueError("low must be <= high for every unit")

    def hook_fn(activation: torch.Tensor, hook: Optional[HookPoint]) -> torch.Tensor:
        d = activation.shape[-1]
        for name, t in (("low", low_t), ("high", high_t), ("value", value_t)):
            if t.ndim == 1 and t.shape[0] != d:
                raise ValueError(f"{name} has {t.shape[0]} units but the activation has {d}")
        # Compare in at least float32 so thresholds are not rounded to a bf16/fp16 grid.
        dtype = torch.promote_types(activation.dtype, torch.float32)
        dtype = torch.promote_types(dtype, torch.promote_types(low_t.dtype, high_t.dtype))
        x = activation.to(dtype)
        inside = (x >= low_t.to(activation.device, dtype)) & (
            x <= high_t.to(activation.device, dtype)
        )
        fill = value_t.to(device=activation.device, dtype=activation.dtype)
        return torch.where(inside, fill, activation)

    return hook_fn
