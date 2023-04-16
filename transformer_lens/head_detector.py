import torch
import numpy as np
from transformer_lens import HookedTransformer, ActivationCache
from typing import List, Tuple

HEAD_NAMES = ['previous_token_head', 'duplicate_token_head', 'induction_head']

def detect_head(model: HookedTransformer, 
                                sequence: str,
                                head_name: str = None,
                                detection_pattern: torch.Tensor = None, 
                                specific_heads: List[Tuple[int, int]] = None,
                                cache: ActivationCache = None,
                                exclude_bos: bool = False,
                                exclude_current_token: bool = False) -> torch.Tensor:
  
    """Searches the model (or a set of specific heads, for circuit analysis) for a particular type of attention head. This head is specified
    by a detection pattern, a (sequence_length, sequence_length) tensor representing how much attention to keep at each position. We element-wise 
    multiply the attention pattern by the detection pattern, and our score is how much attention is left, divided by the total attention. 
    (1 per token other than the first). 
    
    For instance, a perfect previous token head would put 1 attention to the previous token and 0 to everything else. To write a detection pattern
    for this head, we would write a (sequence_length, sequence_length) tensor with 1s below the diagonal and 0s everywhere else. That would then
    be element-wise multiplied by the attention pattern, zeroing out all attention that did not attend to the previous token. The attention
    remaining divided by the total attention would then be our score for that attention head. (This particular head is already implemented in HEAD_NAMES)

    Args:
      model: Model being used.
      sequence: String being fed to the model.
      head_name: Name of an existing head in HEAD_NAMES we want to check. Must pass either a head_name or a detection_pattern, but not both!
      detection_pattern: (sequence_length, sequence_length) Tensor representing what attention pattern corresponds to the head we're looking for.
      specific_heads: If a specific list of heads is given here, all other heads' score is set to -1. Useful for IOI-style circuit analysis.
      cache: Include the cache to save time if you want.
      exclude_bos: Exclude attention paid to the beginning of sequence token.
      exclude_current_token: Exclude attention paid to the current token.

    Returns a (n_layers, n_heads) Tensor representing the score for each attention head.
    
    Example:
    --------
    .. code-block:: python

        >>> from transformer_lens.head_detector import detect_head
        >>> from transformer_lens.HookedTransformer import HookedTransformer
        >>> import plotly.express as px
        >>> import transformer_lens.utils as utils

        >>> def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
        >>>     px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

        >>> model = HookedTransformer.from_pretrained('gpt2-small')
        >>> sequence = "This is a test sequence. This is a test sequence."

        >>> attention_score = detect_head(model, sequence, head_type='previous_token_head')
        >>> imshow(attention_score, zmin=-1, zmax=1, xaxis="Head", yaxis="Layer", title="Previous Head Matches")
    """
    
    assert (head_name is None) != (detection_pattern is None), "Exactly one of head_name or detection_pattern must be specified."
    device = model.cfg.device

    if head_name is not None:
      assert head_name in HEAD_NAMES, "Head name not valid."
      detection_pattern = eval(f'get_{head_name}_detection_pattern(model, sequence)').to(device)

    sequence = model.to_tokens(sequence).to(device)

    assert 1 < sequence.shape[-1] < model.cfg.n_ctx, "The sequence must be non-empty and must fit within the model's context window."
    assert sequence.shape[-1] == detection_pattern.shape[0] == detection_pattern.shape[1], "The detection pattern must be a square matrix of shape (sequence_length, sequence_length)."

    if cache is None:
        _, cache = model.run_with_cache(sequence, remove_batch_dim=True)

    heads = [(i, j) for i in range(model.cfg.n_layers) for j in range(model.cfg.n_heads)]

    # If no specific heads are mentioned, grab all of them.
    if specific_heads is None:
        specific_heads = [(i, j) for i in range(model.cfg.n_layers) for j in range(model.cfg.n_heads)]

    matches = []

    for head in heads:
        if head not in specific_heads:
            matches.append(-1) # This is kinda ugly, the idea is to be able to still plot this on a 2D grid and everything not in the circuit is red. Ideas?
        else:
            attention_pattern = cache["pattern", head[0], "attn"]
            attention_pattern = attention_pattern[head[1], :, :] # We could batch this, but it only takes a few seconds right now. Can worry about that later. 
            # It'd be a bit more complex of a batch when using specific heads. Could also just batch-compute all heads if it's faster, then set the non-matching heads to -1 afterwards.
            
            if exclude_bos:
              attention_pattern[:, 0] = 0

            if exclude_current_token:
              attention_pattern.fill_diagonal_(0)

            # Elementwise multiplication keeps only the parts of the attention pattern that are also in the detection pattern.
            matched_attention = attention_pattern * detection_pattern
            matches.append((torch.sum(matched_attention) / torch.sum(attention_pattern))) # Return total percentage of attention which matches the detection pattern.

    matches = torch.as_tensor(matches)
    return matches.reshape(model.cfg.n_layers, model.cfg.n_heads)

# Previous token head
def get_previous_token_head_detection_pattern(model: HookedTransformer, sequence: str) -> torch.Tensor:
    """Outputs a detection score for previous token heads. 
    Previous token head: https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=0O5VOHe9xeZn8Ertywkh7ioc
    
    Args:
      model: Model being used.
      sequence: String being fed to the model."""
    
    sequence = model.to_tokens(sequence)
    detection_pattern = torch.zeros((sequence.shape[-1], sequence.shape[-1]))
    detection_pattern[1:, :-1] = torch.eye(sequence.shape[-1] - 1) # Adds a diagonal of 1's below the main diagonal.
    return torch.tril(detection_pattern)

# Duplicate token head
def get_duplicate_token_head_detection_pattern(model: HookedTransformer, sequence: str) -> torch.Tensor:
    """Outputs a detection score for duplicate token heads. 
    Duplicate token head: https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=2UkvedzOnghL5UHUgVhROxeo
    
    Args:
      model: Model being used.
      sequence: String being fed to the model."""
    
    sequence = model.to_tokens(sequence).detach().cpu()

    # Repeat sequence to create a square matrix.
    token_pattern = sequence.repeat(sequence.shape[-1], 1).numpy()

    # If token_pattern[i][j] matches its transpose, then token j and token i are duplicates.
    eq_mask = np.equal(token_pattern, token_pattern.T).astype(int)

    np.fill_diagonal(eq_mask, 0) # Current token is always a duplicate of itself. Ignore that.
    detection_pattern = eq_mask.astype(int)
    return torch.tril(torch.as_tensor(detection_pattern).float())

# Induction head
def get_induction_head_detection_pattern(model: HookedTransformer, sequence: str) -> torch.Tensor:
    """Outputs a detection score for induction heads. 
    Induction head: https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=_tFVuP5csv5ORIthmqwj0gSY
    
    Args:
      model: Model being used.
      sequence: String being fed to the model."""
    
    duplicate_pattern = get_duplicate_token_head_detection_pattern(model, sequence)

    # Shift all items one to the right
    shifted_tensor = torch.roll(duplicate_pattern, shifts=1, dims=1)

    # Replace first column with 0's - we don't care about bos but shifting to the right moves the last column to the first, and the last column might contain non-zero values.
    zeros_column = torch.zeros((duplicate_pattern.shape[0], 1))
    result_tensor = torch.cat((zeros_column, shifted_tensor[:, 1:]), dim=1)
    return torch.tril(result_tensor)

def get_supported_heads():
    """Returns a list of supported heads."""
    print(f"Supported heads: {HEAD_NAMES}")