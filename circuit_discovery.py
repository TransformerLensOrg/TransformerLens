# %% [markdown]
# Imports

from typing import List, Tuple, Dict, Union, Optional, Callable, Any
from time import ctime
import torch
import numpy as np
from copy import deepcopy
from collections import OrderedDict
from easy_transformer.ioi_dataset import IOIDataset
import pickle
from subprocess import call

from easy_transformer import EasyTransformer

from easy_transformer.ioi_dataset import (
    IOIDataset,
)
from easy_transformer.utils_circuit_discovery import (
    evaluate_circuit,
    direct_path_patching,
    logit_diff_io_s,
    HypothesisTree,
    logit_diff_from_logits,
    get_datasets,
)

from easy_transformer.ioi_utils import (
    show_pp,
)

from IPython import get_ipython

ipython = get_ipython()
if ipython is not None:
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")

file_prefix = "pngs/" if os.path.exists("pngs") else ""

#%% [markdown]
# Load in the model

model_name = "gpt2"  # @param ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'facebook/opt-125m', 'facebook/opt-1.3b', 'facebook/opt-2.7b', 'facebook/opt-6.7b', 'facebook/opt-13b', 'facebook/opt-30b', 'facebook/opt-66b', 'EleutherAI/gpt-neo-125M', 'EleutherAI/gpt-neo-1.3B', 'EleutherAI/gpt-neo-2.7B', 'EleutherAI/gpt-j-6B', 'EleutherAI/gpt-neox-20b']

model = EasyTransformer.from_pretrained(model_name)
model.set_use_attn_result(True)
model.set_use_headwise_qkv_input(
    True
)  # this is an extra option, that allows us to control the Q, K and V inputs to the attention heads

#%% [markdown]
# # Load data

N = 50

dataset_orig = IOIDataset(
    prompt_type="mixed",
    N=N,
    tokenizer=model.tokenizer,
    prepend_bos=False,
)

# baseline dataset
dataset_new = (
    dataset_orig.gen_flipped_prompts(("IO", "RAND"))
    .gen_flipped_prompts(("S", "RAND"))
    .gen_flipped_prompts(("S1", "RAND"))
)

print(
    f"These dataset objects hold labels to all the relevant words in the sentences: {dataset_orig.word_idx.keys()}"
)

#%% [markdown]
# Get the initial logit difference

model.reset_hooks()
logit_diff_initial = logit_diff_io_s(model, dataset_orig)
print(f"Initial logit difference: {logit_diff_initial:.3f}")

#%% [markdown]
# Simplest path patching run

receivers_to_senders = {
    ("blocks.11.hook_resid_post", None): [
        ("blocks.9.attn.hook_result", 9, "end"),
        ("blocks.10.attn.hook_result", 0, "end"),
        ("blocks.9.attn.hook_result", 6, "end"),
    ]
}

# the IOI paper claims that heads 9.9, 10.0, 9.6 are the most important heads for writing to the residual stream
# the above object specifies that we should patch the three edges from these heads to the end state of the residual stream
# the string literals will become familiar after learning https://github.com/neelnanda-io/Easy-Transformer/blob/main/EasyTransformer_Demo.ipynb

#%%
# Now do the direct path patching

model = direct_path_patching(  # direct path patching returns a model with attached hooks that are relevant for the patch
    model=model,
    orig_data=dataset_orig.toks.long(),
    new_data=dataset_new.toks.long(),  # all hooks that aren't senders will be patched to new_data
    receivers_to_senders=receivers_to_senders,
    orig_positions=dataset_orig.word_idx,
    new_positions=dataset_new.word_idx,
)

new_logit_diff = logit_diff_io_s(model, dataset_orig)
print(f"New logit difference: {new_logit_diff:.3f}")  # this should be lower

#%% [markdown]
# Do the most complex run

# the hooks ...hook_k_input (and q_input, v_input) allow editing of the Q, K and V inputs to the attention heads
# the hooks ...hook_resid_mid allow editing of the input to MLPs
# the hook blocks.0.hook_resid_pre allows editing from the embeddings

receivers_to_senders = {
    ("blocks.11.hook_resid_post", None): [
        ("blocks.9.attn.hook_result", 9, "end"),
    ],
    ("blocks.9.attn.hook_k_input", 9): [
        ("blocks.0.hook_mlp_out", None, "IO"),
    ],
    ("blocks.0.hook_resid_mid", None): [
        ("blocks.0.hook_resid_pre", None, "IO"),
    ],
}

# Now do the direct path patching
model = direct_path_patching(
    model=model,
    orig_data=dataset_orig.toks.long(),
    new_data=dataset_new.toks.long(),
    receivers_to_senders=receivers_to_senders,
    orig_positions=dataset_orig.word_idx,
    new_positions=dataset_new.word_idx,
)

ans = logit_diff_io_s(model, dataset_orig)
model.reset_hooks()
print(f"{ans=}")
print(f"{logit_diff_initial=}, {ans=} (this difference should be small but not 0")
assert np.abs(logit_diff_initial - ans) > 1e-9, "!!!"

#%% [markdown]
# Patch patching

model.reset_hooks()

# construct the position labels
orig_positions = OrderedDict()
new_positions = OrderedDict()
keys = ["IO", "S+1", "S", "S2", "end"]
for key in keys:
    orig_positions[key] = dataset_orig.word_idx[key]
    new_positions[key] = dataset_new.word_idx[key]

# make the tree object
h = HypothesisTree(
    model,
    metric=logit_diff_io_s,
    dataset=dataset_orig,  # metric is a function of the hooked model and the dataset, so keep context about dataset_orig inside the dataset object
    orig_data=dataset_orig.toks.long(),
    new_data=dataset_new.toks.long(),
    threshold=0.05,
    orig_positions=orig_positions,
    new_positions=new_positions,
    use_caching=True,
    direct_paths_only=True,
)

#%% [markdown]
# Run path patching

while h.current_node is not None:
    h.eval(show_graphics=False, verbose=True)
    a = h.show()

    # save digraph object
    with open("hypothesis_tree.dot", "w") as f:
        f.write(a.source)

    # convert to png
    call(
        [
            "dot",
            "-Tpng",
            "hypothesis_tree.dot",
            "-o",
            f"gpt2_hypothesis_tree_{ctime()}.png",
            "-Gdpi=600",
        ]
    )
#%% [markdown]
# What about if we run the circuit on the original data ONLY at the nodes in the graph?

evaluate_circuit(h, dataset_new)  # close to 3, but still missing some parts

#%% [markdown]
# Try this on a new dataset

template = "Yesterday it was{day} so today it is"
all_days = [
    " Monday",
    " Tuesday",
    " Wednesday",
    " Thursday",
    " Friday",
    " Saturday",
    " Sunday",
]
sentences = []
answers = []
wrongs = []

for day_idx in range(7):
    cur_sentence = template.format(day=all_days[(day_idx + 1) % 7])
    cur_ans = all_days[day_idx]
    cur_correct_index = model.tokenizer
    sentences.append(cur_sentence)
    answers.append(cur_ans)
    wrongs.append(all_days[day_idx])

tokens = model.to_tokens(sentences, prepend_bos=True)
answers = torch.tensor(model.tokenizer(answers)["input_ids"]).unsqueeze(
    -1
)  # , prepend_bos=True)
wrongs = torch.tensor(model.tokenizer(wrongs)["input_ids"]).unsqueeze(
    -1
)  # , prepend_bos=True)

positions = OrderedDict()
positions["Yesterday"] = torch.ones(size=(7,))
positions["Day"] = torch.ones(size=(7,)) * 4
positions["Today"] = torch.ones(size=(7,)) * 6


def day_metric(model, dataset):
    logits = model(tokens)
    logits_on_correct = logits[torch.arange(7), answers]
    logits_on_wrong = logits[torch.arange(7), wrongs]
    return (logits_on_correct - logits_on_wrong).mean()


fake_data = tokens.clone()
tokens[:, 1] = model.to_tokens("Earlier", prepend_bos=False).item()
tokens[:, 4] = model.to_tokens(" hot", prepend_bos=False).item()
tokens[:, 5] = model.to_tokens(" but", prepend_bos=False).item()
tokens[:, 6] = model.to_tokens(" now", prepend_bos=False).item()
# "Earlier it was hot but now it is"

h = HypothesisTree(
    model,
    metric=day_metric,
    dataset=None,  # metric is a function of the hooked model and the dataset, so keep context about dataset_orig inside the dataset object
    orig_data=tokens,
    new_data=fake_data,
    threshold=0.05,
    orig_positions=positions,
    new_positions=positions,
    use_caching=True,
    direct_paths_only=True,
)
