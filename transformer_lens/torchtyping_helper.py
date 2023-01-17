from enum import Enum

class T(Enum):
    """Helper class to get mypy to work with TorchTyping and solidify naming conventions as a byproduct.
    
    Examples: 
    - `TT[T.batch, T.pos, T.d_model]`
    - `TT[T.num_components, T.batch_and_pos_dims:...]`
    - `TT[T.d_vocab + T.n_ctx, T.d_model]`
    """

    batch = "batch"
    pos = "pos"
    head_index = "head_index"
    length = "length"
    rotary_dim = "rotary_dim"
    new_tokens = "new_tokens"
    batch_and_pos_dims = "batch_and_pos_dims"
    layers_accumulated_over = "layers_accumulated_over"
    layers_covered = "layers_covered"
    past_kv_pos_offset = "past_kv_pos_offset"
    num_components = "num_components"
    num_neurons = "num_neurons"
    pos_so_far = "pos_so_far"
    n_ctx = "n_ctx"
    n_heads = "n_heads"
    n_layers = "n_layers"
    d_vocab = "d_vocab"
    d_vocab_out = "d_vocab_out"
    d_head = "d_head"
    d_mlp = "d_mlp"
    d_model = "d_model"

    ldim = "ldim"
    rdim = "rdim"
    new_rdim = "new_rdim"
    mdim = "mdim"
    leading_dims = "leading_dims"
    leading_dims_left = "leading_dims_left"
    leading_dims_right = "leading_dims_right"

    a = "a"
    b = "b"

    def __add__(self, other: "T") -> str:
        """Hack to let us write type expressions like `T.a + T.b`"""
        return f"{self.value} + {other.value}"