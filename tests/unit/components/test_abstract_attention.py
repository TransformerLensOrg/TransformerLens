import torch

from transformer_lens.components import AbstractAttention


def test_create_alibi_slope():
    n_ctx = 100

    # Expected result computed non-vectorized way
    expected = torch.zeros((n_ctx, n_ctx))
    for row in range(n_ctx):
        for col in range(n_ctx):
            expected[row, col] = float(min(col - row, 0))

    # Check against the method's vectorized version
    result = AbstractAttention.create_alibi_slope(n_ctx)
    assert torch.allclose(expected, result)


def test_create_alibi_bias():
    n_heads = 2
    n_ctx = 4

    result = AbstractAttention.create_alibi_bias(n_heads, n_ctx, torch.device("cpu"))

    for matrix in result:
        n_row, n_col = matrix.size()
        slope = -matrix[1, 0]
        # Check if upper triangle is all zeros
        assert torch.equal(torch.triu(matrix), torch.zeros_like(matrix))

        ref_lower_triangle = torch.zeros_like(matrix)
        for i in range(1, n_row):
            for j in range(i):
                ref_lower_triangle[i, j] = -slope * (i - j)

        # Check if the lower triangle is decreasing by a constant slope (towards the bottom left corner).
        assert torch.equal(
            torch.tril(matrix, diagonal=-1), torch.tril(ref_lower_triangle, diagonal=-1)
        )
