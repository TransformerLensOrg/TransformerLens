#%% [markdown]
# <p>This notebook covers the creation of an automatic circuit discovery experiment, and detailed explanation of how to generate automatic circuit pictures for the IOI task</p>

#%% [markdown]
# <h3>Sort out whether we're in a notebook or not</h3>

import os

try:
    import google.colab
    IN_COLAB = True
    print("Running as a Colab notebook")
    os.system("pip install git+https://github.com/ArthurConmy/Easy-Transformer.git")

except:
    IN_COLAB = False
    print("Running as a Jupyter notebook - intended for development only!")

# %% [markdown]
# <h2>Imports</h2>

from typing import List, Tuple, Dict, Union, Optional, Callable, Any
from time import ctime
import einops
import torch
import numpy as np
from copy import deepcopy
from collections import OrderedDict
import pickle
from subprocess import call
from IPython import get_ipython

ipython = get_ipython()
if ipython is not None:
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")
from easy_transformer import EasyTransformer
from easy_transformer.utils_circuit_discovery import (
    evaluate_circuit,
    patch_all,
    direct_path_patching,
    logit_diff_io_s,
    Circuit,
    path_patching,
    logit_diff_from_logits,
    get_datasets,
)
from easy_transformer.experiments import (
    get_act_hook,
)
from easy_transformer.ioi_utils import (
    show_pp,
)
from easy_transformer.ioi_dataset import IOIDataset
import os

file_prefix = "archive/" if os.path.exists("archive") else ""

#%% [markdown]
# <h2>Load in the model</h2>

model_name = "gpt2" # @param ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'facebook/opt-125m', 'facebook/opt-1.3b', 'facebook/opt-2.7b', 'facebook/opt-6.7b', 'facebook/opt-13b', 'facebook/opt-30b', 'facebook/opt-66b', 'EleutherAI/gpt-neo-125M', 'EleutherAI/gpt-neo-1.3B', 'EleutherAI/gpt-neo-2.7B', 'EleutherAI/gpt-j-6B', 'EleutherAI/gpt-neox-20b']
model = EasyTransformer.from_pretrained(model_name)

#%% [markdown]
# <h2>Make the dataset</h2>

template = "Last month it was {month} so this month it is"
all_months = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
sentences = []
answers = []
wrongs = []
batch_size = 12
for month_idx in range(batch_size):
    cur_sentence = template.format(month=all_months[month_idx])
    cur_ans = all_months[(month_idx + 1) % batch_size]
    sentences.append(cur_sentence)
    answers.append(cur_ans)
    wrongs.append(all_months[month_idx])

tokens = model.to_tokens(sentences, prepend_bos=True)
answers = torch.tensor(model.tokenizer(answers)["input_ids"]).squeeze()
wrongs = torch.tensor(model.tokenizer(wrongs)["input_ids"]).squeeze()

#%% [markdown]
# <h3>Make the positions labels (step 1)</h3>

positions = OrderedDict()
ones = torch.ones(size=(batch_size,)).long()
positions["Last"] = ones.clone()
positions["word month"] = ones.clone() * 2
positions["month"] = ones.clone() * 5
positions["word month 2"] = ones.clone() * 8
positions["END"] = ones.clone() * 10

#%% [markdown]
# <h3>Make the baseline dataset (step 2)</h3>

baseline_data = tokens.clone()
baseline_data[0] = model.to_tokens("This time it is here and last time it was", prepend_bos=True)
baseline_data = einops.repeat(baseline_data[0], "s -> b s", b=baseline_data.shape[0])

#%% [markdown]
# <h3>Define the metric (step 3)</h3>

def day_metric(model, dataset):
    logits = model(tokens)
    logits_on_correct = logits[torch.arange(batch_size), -1, answers]
    logits_on_wrong = logits[torch.arange(batch_size), -1, wrongs]
    ans = torch.mean(logits_on_correct - logits_on_wrong)
    return ans.item()

#%% [markdown]
# Make the circuit object

h = Circuit(
    model,
    metric=day_metric,
    orig_data=tokens,
    new_data=baseline_data,
    threshold=0.25,
    orig_positions=positions,
    new_positions=positions, # in some datasets we might want to patch from different positions; not here
)
#%% [markdown]
# <h2> Run path patching! </h2>
# <p> Only the first two lines of this cell matter; the rest are for saving images. This cell takes several minutes to run. If you cancel and then call h.show(), you can see intermediate representations of the circuit. </p>

while h.current_node is not None:
    h.eval(show_graphics=False, verbose=True)

    a = h.show()
    # save digraph object
    with open(file_prefix + "hypothesis_tree.dot", "w") as f:
        f.write(a.source)

    # convert to png
    call(
        [
            "dot",
            "-Tpng",
            "hypothesis_tree.dot",
            "-o",
            file_prefix + f"gpt2_hypothesis_tree_{ctime()}.png",
            "-Gdpi=600",
        ]
    )
#%% [markdown]
# <h2> Show the circuit </h2>
h.show()

#%% [markdown]
# <h2>What about if we run the circuit on the original data ONLY at the nodes in the graph?</h2>
evaluate_circuit(h, None) # positive, but very small - we've likely missed some indices. Project: find which ones!

#%% [markdown]
# <h1>IOI Patching</h1>
# <p>The rest of this notebook covers how the direct_path_patching function works internally, using the IOI dataset as an example</p>

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
# <h2>Get the initial logit difference</h2>

model.reset_hooks()
logit_diff_initial = logit_diff_io_s(model, dataset_orig)
print(f"Initial logit difference: {logit_diff_initial:.3f}")

#%% [markdown]
# <h2>Simplest path patching run</h2>

receivers_to_senders = {
    ("blocks.11.hook_resid_post", None): [
        ("blocks.9.attn.hook_result", 9, "end"),
        ("blocks.10.attn.hook_result", 0, "end"),
        ("blocks.9.attn.hook_result", 6, "end"),
        # ("blocks.11.attn.hook_result", 10, "end"),
    ]
}

# the IOI paper claims that heads 9.9, 10.0, 9.6 are the most important heads for writing to the residual stream
# the above object specifies that we should patch the three edges from these heads to the end state of the residual stream
# the string literals will become familiar after learning https://github.com/neelnanda-io/Easy-Transformer/blob/main/EasyTransformer_Demo.ipynb

#%%
# <h2>Now do the direct path patching</h2>

model = direct_path_patching(  # direct path patching returns a model with attached hooks that are relevant for the patch
    model=model,
    orig_data=dataset_orig.toks.long(),
    new_data=dataset_new.toks.long(),  # all hooks that aren't senders will be patched to new_data
    receivers_to_senders=receivers_to_senders,
    orig_positions=dataset_orig.word_idx,
    new_positions=dataset_new.word_idx,
)

new_logit_diff = logit_diff_io_s(model, dataset_orig)
print(f"New logit difference: {new_logit_diff:.3f}")  # this should be negative: without these heads, the model can't distinguish between IO and S!

#%% [markdown]
# <h2>Do direct_path_patching with all possible features</h2>
# <p>the hooks ...hook_k_input (and q_input, v_input) allow editing of the Q, K and V inputs to the attention heads
# the hooks ...hook_resid_mid allow editing of the input to MLPs
# the hook blocks.0.hook_resid_pre allows editing from the embeddings</p>

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
print(f"{logit_diff_initial=}, {ans=} (this difference should be small but not 0)")
assert np.abs(logit_diff_initial - ans) > 1e-9, "!!!"

#%% [markdown]
# <h2>Automatic circuit discovery</h2>

model.reset_hooks()

# construct the position labels
orig_positions = OrderedDict()
new_positions = OrderedDict()
keys = ["IO", "S+1", "S", "S2", "end"]
for key in keys:
    orig_positions[key] = dataset_orig.word_idx[key]
    new_positions[key] = dataset_new.word_idx[key]

# make the tree object
h = Circuit(
    model,
    metric=logit_diff_io_s,
    dataset=dataset_orig,  # metric is a function of the hooked model and the dataset, so keep context about dataset_orig inside the dataset object
    orig_data=dataset_orig.toks.long(),
    new_data=dataset_new.toks.long(),
    threshold=0.25,
    orig_positions=orig_positions,
    new_positions=new_positions,
    use_caching=True,
)

#%% [markdown]
# <h2>Run circuit discovery</h2>

while h.current_node is not None:
    h.eval(show_graphics=True, verbose=True)

    a = h.show()
    # save digraph object
    with open(file_prefix + "hypothesis_tree.dot", "w") as f:
        f.write(a.source)

    # convert to png
    call(
        [
            "dot",
            "-Tpng",
            "hypothesis_tree.dot",
            "-o",
            file_prefix + f"gpt2_hypothesis_tree_{ctime()}.png",
            "-Gdpi=600",
        ]
    )