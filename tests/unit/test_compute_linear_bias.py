# import pytest
# import torch

# from transformer_lens import HookedTransformer


# @pytest.fixture
# def attention_instance():
#     ht = HookedTransformer.from_pretrained("bloom-560m")
#     attention = ht.blocks[0].attn
#     return attention


# def test_create_alibi_slope(attention_instance):
#     pass
# n_ctx = 100

# # Expected result computed non-vectorized way
# expected = torch.zeros((n_ctx, n_ctx))
# for row in range(n_ctx):
#     for col in range(n_ctx):
#         expected[row, col] = float(min(col - row, 0))

# # Check against the method's vectorized version
# result = attention_instance.create_alibi_slope(n_ctx)
# assert torch.allclose(expected, result)


# def test_create_alibi_multipliers(attention_instance):
#     n_heads = 8
#     start = 2 ** (-8 / n_heads)
#     indices = torch.arange(n_heads)
#     expected = start * (start**indices)

#     result = attention_instance.create_alibi_multipliers(n_heads)
#     assert torch.allclose(expected, result)


# def test_create_alibi_bias(attention_instance):
#     n_heads = 2
#     n_ctx = 4

#     result = attention_instance.create_alibi_bias(
#         n_heads, n_ctx, attention_instance.cfg.device
#     )
#     for matrix in result:
#         n_row, n_col = matrix.size()
#         slope = -matrix[1, 0]
#         # Check if upper triangle is all zeros
#         assert torch.equal(torch.triu(matrix), torch.zeros_like(matrix))

#         ref_lower_triangle = torch.zeros_like(matrix)
#         for i in range(1, n_row):
#             for j in range(i):
#                 ref_lower_triangle[i, j] = -slope * (i - j)

#         # Check if the lower triangle is decreasing by a constant slope (towards the bottom left corner).
#         assert torch.equal(
#             torch.tril(matrix, diagonal=-1), torch.tril(ref_lower_triangle, diagonal=-1)
#         )
