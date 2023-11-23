# %%

import torch


def gen_matrix(shape, seed):
    torch.manual_seed(seed)
    return torch.rand(shape)


ays = [gen_matrix((2, 1000), 0) for _ in range(2)]
bees = [gen_matrix((1000, 4), 1) for _ in range(2)]

ays[1] += 1e-10
bees[1] -= 1e-10  # !!

assert (ays[0] - ays[1]).abs().max() < 1e-9  # :)))
assert (bees[0] - bees[1]).abs().max() < 1e-9

prods = [torch.nn.functional.linear(a, b.T) for a, b in zip(ays, bees)]

assert (prods[0] - prods[1]).abs().max() < 1e-9
