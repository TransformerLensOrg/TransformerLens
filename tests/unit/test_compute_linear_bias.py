import torch

from transformer_lens.utils import expand_alibi_on_query_dim


def test_expand_alibi_on_query_dim():
    # Define the tensor x
    x = torch.tensor([[[[1, 2, 3, 4, 5, 6, 7]]]])  # shape: (1, 1, 1, 7)
    query_pos = 7
    result = expand_alibi_on_query_dim(x, query_pos)

    for i in range(query_pos):
        # The first i values should be decreasing from 7 to 7-i+1
        expected_first_values = torch.arange(7 - i + 1, 8, device=x.device)

        # The remaining values should be 7
        expected_remaining_values = torch.tensor(
            [7] * (query_pos - len(expected_first_values)), device=x.device
        )

        # Concatenate to form the expected row
        expected_row = torch.cat((expected_first_values, expected_remaining_values))

        # Check if the result matches the expected row
        if not torch.equal(result[0, 0, i], expected_row):
            return False

    return True


def test_diagonal_equal():
    n_heads = 16
    key_pos = 50
    batch_size = 20
    query_pos = 50
    x = torch.rand(n_heads, 1, key_pos)
    x = x.repeat(batch_size, 1, 1, 1)
    alibi = expand_alibi_on_query_dim(x, query_pos)
    assert alibi.shape == (batch_size, n_heads, query_pos, key_pos)

    diagonals = torch.diagonal(alibi, dim1=-2, dim2=-1)

    # Check if each diagonal has only one unique value
    unique_counts = diagonals.unique(dim=-1).size(-1)
    assert unique_counts == 1
