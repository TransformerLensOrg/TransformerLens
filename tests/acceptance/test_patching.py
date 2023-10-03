import pytest 

import torch
from transformer_lens.HookedTransformer import HookedTransformer
from transformer_lens.patching import get_act_patch_attn_out


@pytest.fixture(scope="module")
def model():
    return HookedTransformer.from_pretrained("gpt2-small")


@pytest.fixture(scope="module")
def clean_input():
    
    name1 = 'Bob'
    name2 = 'Carl'
    object = 'milk'
    sentence = f"{name1} met with {name2}. {name2} gave the {object} to" # expect name 1
    return sentence

@pytest.fixture(scope="module")
def corrupt_input():
    
    name1 = 'Bob'
    name2 = 'Carl'
    object = 'milk'
    sentence = f"{name1} met with {name2}. {name1} gave the {object} to" # expect name 2

    return sentence

def test_get_act_patch_attn_out(model, corrupt_input, clean_input):
    
    # tokenize the corrupted tokens. 
    clean_tokens = model.tokenizer.encode(clean_input)
    corrupt_tokens = torch.tensor(model.tokenizer.encode(corrupt_input)).unsqueeze(0)
    
    # need to run with cache on clean to get clean cache. 
    logits, clean_cache = model.run_with_cache(
        clean_input
    )
    
    
    # patching metric will be logit difference, so we'll need the indexes
    # we assume the indexes for name1 is 0 and name2 is -1 in corrupt tokens
    idx_name1 = 18861 # Bob 
    idx_name2 = 26886 # Carl
    
    
    def patching_metric(logits):
        """
        Linear function of logit diff, calibrated so that it equals 0 when performance is
        same as on corrupted input, and 1 when performance is same as on clean input.
        """

        logit_diff = logits[0, -1, idx_name1] - logits[0, -1, idx_name2]
        return logit_diff


    scan = get_act_patch_attn_out(
        model = model,
        corrupted_tokens=corrupt_tokens, 
        clean_cache=clean_cache,
        patching_metric=patching_metric
    )
    
    assert scan.shape == [12,10] # GPT2 has 12 layers, there are 10 attention heads

    # Plotting this 
    # px.imshow(scan, color_continuous_scale="RdBu", color_continuous_midpoint=0).show()
