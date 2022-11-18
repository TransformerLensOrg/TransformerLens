# %%
import os
import torch

if os.environ["USER"] in ["exx", "arthur"]:  # so Arthur can safely use octobox
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
assert torch.cuda.device_count() == 1
from easy_transformer.EasyTransformer import LayerNormPre
from tqdm import tqdm
from copy import deepcopy
import pandas as pd
import torch
import torch as t
from easy_transformer.utils import (
    gelu_new,
    to_numpy,
    get_corner,
    print_gpu_mem,
)  # helper functions
from easy_transformer.hook_points import HookedRootModule, HookPoint
from easy_transformer.EasyTransformer import (
    EasyTransformer,
    TransformerBlock,
    MLP,
    Attention,
    LayerNormPre,
    PosEmbed,
    Unembed,
    Embed,
)
from easy_transformer.experiments import (
    ExperimentMetric,
    AblationConfig,
    EasyAblation,
    EasyPatching,
    PatchingConfig,
    get_act_hook,
)
from time import ctime
from functools import partial
from typing import Any, Callable, Dict, List, Set, Tuple, Union, Optional, Iterable
import itertools
import numpy as np
from tqdm import tqdm
import pandas as pd
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly
from sklearn.linear_model import LinearRegression
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import spacy
import re
from einops import rearrange
import einops
from pprint import pprint
import gc
from datasets import load_dataset
from IPython import get_ipython
import matplotlib.pyplot as plt
import random as rd

from easy_transformer.ioi_dataset import (
    IOIDataset,
    NOUNS_DICT,
    NAMES,
    gen_flipped_prompts,
    gen_prompt_uniform,
    BABA_TEMPLATES,
    ABBA_TEMPLATES,
)
from easy_transformer.ioi_utils import (
    all_subsets,
    clear_gpu_mem,
    show_tokens,
    show_pp,
    show_attention_patterns,
    safe_del,
)
from easy_transformer.ioi_circuit_extraction import (
    do_circuit_extraction,
    gen_prompt_uniform,
    get_act_hook,
    get_circuit_replacement_hook,
    get_extracted_idx,
    get_heads_circuit,
    join_lists,
    list_diff,
    process_heads_and_mlps,
    turn_keep_into_rmv,
    CIRCUIT,
)

from rich import print as rprint

ipython = get_ipython()
if ipython is not None:
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")

#%% [markdown]
# Load the model

model_name = "gpt2"  # Here we used gpt-2 small ("gpt2")
print_gpu_mem("About to load model")
model = EasyTransformer.from_pretrained(
    model_name,
)
model.set_use_attn_result(True)
device = "cuda"
if torch.cuda.is_available():
    model.to(device)
print_gpu_mem("Gpt2 loaded")
#%%
N = 150
ioi_dataset = IOIDataset(prompt_type="mixed", N=N, tokenizer=model.tokenizer)

#%%


def test_prompt(prompt, answer, prepend_space_to_answer=False, print_details=True):
    # GPT-2 often treats the first token weirdly, so lets give it a resting position
    prompt = prompt
    if prepend_space_to_answer:
        answer = " " + answer
    # Hacky function to map a text string to separate tokens as text
    prompt_str_tokens = model.tokenizer.batch_decode(model.tokenizer.encode(prompt))
    answer_str_tokens = model.tokenizer.batch_decode(model.tokenizer.encode(answer))
    answer_tokens = model.tokenizer.encode(answer)
    if print_details:
        print("Tokenized prompt:", prompt_str_tokens)
        print("Tokenized answer:", answer_str_tokens)
    prompt_length = len(prompt_str_tokens)
    answer_length = len(answer_str_tokens)
    logits = model(prompt + answer)
    final_print = []
    for index in range(prompt_length, prompt_length + answer_length):
        answer_index = index - prompt_length
        answer_token = answer_tokens[answer_index]
        token_logits = logits[0, index - 1]
        probs = torch.nn.functional.softmax(token_logits, dim=-1)
        values, indices = token_logits.sort(descending=True)
        correct_rank = torch.arange(len(indices))[indices == answer_token].item()
        final_print.append((answer_str_tokens[answer_index], correct_rank))
        if print_details:
            rprint(
                f"[b]Performance on correct token |{answer_str_tokens[answer_index]}|: [/b]"
            )
            rprint(
                f"[b]Rank: {correct_rank}[/b] Logit: {token_logits[answer_token].item():.6} Prob: {probs[answer_token].item():.2%}"
            )
            print()
            for i in range(10):
                rprint(
                    f"Top {i}th logit. Logit: {values[i].item():.6} Prob: {probs[indices[i]].item():.2%} Token: |{model.tokenizer.decode(indices[i])}|"
                )
            print()
    rprint("[b]Ranks of the answer tokens:[/b]", final_print)


# %%

test_prompt(
    """contain: CONTAIN
few: FEW
dead: DEAD
remove: REMOVE
interview: INTERVIEW
carry: CARRY
for: FOR
role: ROLE
might: MIGHT
choice: CHOICE
song: SONG
serious: SERIOUS
structure: STRUCTURE
front: FRONT
nearly: NEARLY
doctor: DOCTOR
foot: FOOT
allow: ALLOW
exactly: EXACTLY
blue: BLUE""",
    " ",
)


test_prompt(
    """Then, Kyle and Jessica were thinking about going to the office. Jessica had a good day. Kyle wanted to give a computer to Jessica.
""",
    " ",
)


# %%
test_prompt(
    """Then, Kyle and Jessica were thinking about going to the office. Jessica had a good day. Kyle wanted to give a computer to Jessica.
    THEN, KYLE AND JESSICA WERE THINKING ABOUT GOING TO THE OFFICE. JESSICA""",
    " ",
)
# %% Run automatic attacks

from easy_transformer.ioi_utils import probs, logit_diff

ADX_TEMPLATE = [
    " [A] had a good day.",
    " [A] was enjoying the situation.",
    " [A] was tired.",
    " [A] enjoyed being with a friend.",
    " [A] was an enthusiast person.",
]

# concatenate every pair from ADX_TEMPLATE
DOUBLE_ADX_TEMPLATE = [x1 + x2 for x1 in ADX_TEMPLATE for x2 in ADX_TEMPLATE]


def gen_adv(ioi_dataset, model, templates, name="IO"):
    adv_ioi_dataset = deepcopy(ioi_dataset)
    for i, s in enumerate(ioi_dataset.sentences):
        adv_temp = rd.choice(templates)
        adv_temp = adv_temp.replace("[A]", ioi_dataset.ioi_prompts[i][name])
        adv_tok_len = len(show_tokens(adv_temp, model, return_list=True))

        punct_idx = int(ioi_dataset.word_idx["punct"][i])
        txt_toks = show_tokens(s, model, return_list=True)
        punct_str_idx = len("".join(txt_toks[:punct_idx]))
        assert (
            s[punct_str_idx] == "." or s[punct_str_idx] == ","
        ), f"{s} --- {s[punct_str_idx]} -- {punct_str_idx} -- {i}"
        s = s[: punct_str_idx + 1] + adv_temp + s[punct_str_idx + 1 :]
        adv_ioi_dataset.ioi_prompts[i]["text"] = s
        adv_ioi_dataset.sentences[i] = s
        adv_ioi_dataset.word_idx["end"][i] += adv_tok_len
        adv_ioi_dataset.word_idx["S2"][i] += adv_tok_len
        adv_ioi_dataset.toks = torch.tensor(
            model.tokenizer(adv_ioi_dataset.sentences, padding=True)["input_ids"]
        )
    return adv_ioi_dataset


N = 500
ioi_dataset = IOIDataset(prompt_type="mixed", N=N, tokenizer=model.tokenizer)
adv_dataset = gen_adv(ioi_dataset, model, DOUBLE_ADX_TEMPLATE, name="IO")
# %%
ld = logit_diff(model, ioi_dataset, all=True)
ld_adv = logit_diff(model, adv_dataset, all=True)

prob = probs(model, ioi_dataset, all=True)
prob_adv = probs(model, adv_dataset, all=True)

prob_s = probs(model, ioi_dataset, all=True, type="s")
prob_adv_s = probs(model, adv_dataset, all=True, type="s")
# %% Compute the variation in logit diff and prob between normal and adv

ld_diff = ld_adv / ld - 1
prob_diff = prob_adv / prob - 1
prob_diff_s = prob_adv_s / prob_s - 1


print(f"Example adv sentence: {adv_dataset.sentences[0]}")

print(
    f"Mean logit diff: {ld.mean():.3f} | Adv: {ld_adv.mean():.3f} | Mean relative var.: {ld_diff.mean():.3f} | Init perf {(ld > 0).cpu().numpy().astype(int).mean():.3f} | Success rate: {torch.logical_and(ld_adv < 0, ld > 0).cpu().numpy().astype(int).mean():.3f}"
)
print(
    f"Mean prob IO: {prob.mean():.3f} | Adv IO: {prob_adv.mean():.3f} | Mean relative var.: {prob_diff.mean():.3f}"
)
print(
    f"Mean prob S: {prob_s.mean():.3f} | Adv S: {prob_adv_s.mean():.3f} | Mean relative var.: {(prob_diff_s).mean():.3f}"
)
print(
    f"Ori pref: Mean logit diff: {ld.mean():.3f} | Mean prob IO: {prob.mean():.3f} | Mean prob S: {prob_s.mean():.3f} |  Init perf {(ld > 0).cpu().numpy().astype(int).mean():.3f} "
)
