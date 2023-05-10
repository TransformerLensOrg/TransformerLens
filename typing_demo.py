# %%

import einops
import torch as t
from jaxtyping import Float
from typeguard import typechecked

ZimZam = Float[t.Tensor, "batch feature"]


@typechecked
def test(x: ZimZam) -> ZimZam:
    return einops.rearrange(x, "f b -> f b")


x = t.rand((10000, 1), dtype=t.float32)

test(x)

# what if "batch" and "feature" now take on different values?

x = t.rand((20000, 2), dtype=t.float32)

test(x)

# ah so indeed batch and feature must only be consistent across a single function call

# now what if we repeat the same strings across type definitions?

ZimZam2 = Float[t.Tensor, "batch feature"]


@typechecked
def test2(x: ZimZam2) -> ZimZam:
    return einops.rearrange(x, "f b -> f b")


@typechecked
def test3(x: ZimZam) -> ZimZam2:
    return einops.rearrange(x, "f b -> f b")


test2(x)
test3(x)

# so the right mental model is that the decorators register
# a dictionary whose keys are the dimension names and
# whose values are the sizes. and the values must be consistent
# across a single function call

# now let's watch the type checker fail


@typechecked
def test4(x: ZimZam) -> ZimZam:
    return einops.rearrange(x, "f b -> b f")


# %%
