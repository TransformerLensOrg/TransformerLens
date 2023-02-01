class T:
    """Helper class to get mypy to work with TorchTyping and solidify naming conventions as a byproduct.
    
    Examples: 
    - `TT[T.batch, T.pos, T.d_model]`
    - `TT[T.num_components, T.batch_and_pos_dims:...]`
    """

    batch: str = "batch"
    pos: str = "pos"
    head_index: str = "head_index"
    length: str = "length"
    rotary_dim: str = "rotary_dim"
    new_tokens: str = "new_tokens"
    batch_and_pos_dims: str = "batch_and_pos_dims"
    layers_accumulated_over: str = "layers_accumulated_over"
    layers_covered: str = "layers_covered"
    past_kv_pos_offset: str = "past_kv_pos_offset"
    num_components: str = "num_components"
    num_neurons: str = "num_neurons"
    pos_so_far: str = "pos_so_far"
    n_ctx: str = "n_ctx"
    n_heads: str = "n_heads"
    n_layers: str = "n_layers"
    d_vocab: str = "d_vocab"
    d_vocab_out: str = "d_vocab_out"
    d_head: str = "d_head"
    d_mlp: str = "d_mlp"
    d_model: str = "d_model"

    ldim: str = "ldim"
    rdim: str = "rdim"
    new_rdim: str = "new_rdim"
    mdim: str = "mdim"
    leading_dims: str = "leading_dims"
    leading_dims_left: str = "leading_dims_left"
    leading_dims_right: str = "leading_dims_right"

    a: str = "a"
    b: str = "b"

    pos_plus_past_kv_pos_offset = "pos + past_kv_pos_offset"
    d_vocab_plus_n_ctx = "d_vocab + n_ctx"
    pos_plus_new_tokens = "pos + new_tokens"
