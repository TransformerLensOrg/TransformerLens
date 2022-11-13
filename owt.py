#%% [markdown]
## Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small
# <h1><b>Intro</b></h1>

# This notebook implements all experiments in our paper (which is available on arXiv).

# For background on the task, see the paper.

# Refer to the demo of the <a href="https://github.com/neelnanda-io/Easy-Transformer">Easy-Transformer</a> library here: <a href="https://github.com/neelnanda-io/Easy-Transformer/blob/main/EasyTransformer_Demo.ipynb">demo with ablation and patching</a>.
#
# Reminder of the circuit:
# <img src="https://i.imgur.com/arokEMj.png">
#%% [markdown]
# Setup
from copy import deepcopy
import torch

assert torch.cuda.device_count() == 1
from tqdm import tqdm
import pandas as pd
import torch
import torch as t
from easy_transformer.EasyTransformer import (
    EasyTransformer,
)
from time import ctime
from functools import partial

import numpy as np
from tqdm import tqdm
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import random
import einops
from IPython import get_ipython
from copy import deepcopy
from ioi_dataset import (
    IOIDataset,
)
from ioi_utils import (
    path_patching,
    max_2d,
    CLASS_COLORS,
    show_pp,
    show_attention_patterns,
    scatter_attention_and_contribution,
)
from random import randint as ri
from ioi_circuit_extraction import (
    do_circuit_extraction,
    get_heads_circuit,
    CIRCUIT,
)
from ioi_utils import logit_diff, probs
from ioi_utils import get_top_tokens_and_probs as g

ipython = get_ipython()
if ipython is not None:
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")
#%% [markdown]
# Initialise model (use larger N or fewer templates for no warnings about in-template ablation)

model = EasyTransformer.from_pretrained("gpt2").cuda()
# model.set_use_attn_result(True)

model2 = EasyTransformer.from_pretrained("gpt2-xl").cuda()
# model2.set_use_attn_result(True)

#%% [markdown]
# Initialise dataset
N = 100
ioi_dataset = IOIDataset(
    prompt_type="mixed",
    N=N,
    tokenizer=model.tokenizer,
    prepend_bos=False,
)
#%% [markdown]
# Hello

# %%
from IPython import get_ipython

ipython = get_ipython()

if ipython is not None:
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")

import math
import os
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from functools import partial
import itertools
import pathlib
from pathlib import PurePath as PP
from copy import copy

# F(NM(child_name)).g().get_unique_c(circuit)

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ[
    "XLA_FLAGS"
] = "--xla_force_host_platform_device_count=8"  # Use 8 CPU devices
os.environ["RR_CIRCUITS_REPR_NAME"] = "true"

RRFS_DIR = os.path.expanduser("~/rrfs")
RRFS_INTERP_MODELS_DIR = f"{RRFS_DIR}/interpretability_models_jax/"
os.environ["INTERPRETABILITY_MODELS_DIR"] = os.environ.get(
    "INTERPRETABILITY_MODELS_DIR",
    os.path.expanduser("~/interp_models_jax/")
    if os.path.exists(os.path.expanduser("~/interp_models_jax/"))
    else RRFS_INTERP_MODELS_DIR,
)
# %%
from tqdm import tqdm
from tabulate import tabulate
import numpy as np

np.random.seed(1726)
import attrs
from attrs import frozen
import torch
import torch as t
import einops
import jax
import plotly.express as px
import matplotlib.pyplot as plt

jax.config.update("jax_platform_name", "cpu")

# import matplotlib
# # hmmm
# matplotlib.rcParams["text.parse_math"] = False
# %%

# %%
import os

os.chdir("/home/arthur/unity")
print("User Arthur")
print(os.getcwd())


from interp.circuit.circuit_compiler.settings import WithCompilerSettings
from interp.circuit.circuit_differentiator import CircuitDifferentiator
from interp.circuit.regression_rewrite import (
    get_poly_feat,
    regression_rewrite_sample,
    rewrite_for_solution_coeffs,
)
from interp.tools.data_loading import get_val_seqs
from interp.circuit.batch_transform import BatchTransform
from interp.circuit.function_rewrites import (
    inv_scale_for_log_exp_p_1_approx_fn,
    log_exp_p_1_expectation_approx_rewrite,
    multip_for_log_exp_p_1_approx_fn,
    pair_log_softmax_to_elementwise,
    softmax_sigmoid_rewrite,
)
from interp.circuit.scope_manager import ScopeManager
from interp.plotting.plot_text_numbers import plot_text_numbers
from interp.circuit import computational_node
from interp.circuit.sampling_estimation import (
    SamplingInstance,
    get_make_default_eval_many,
)
from interp.circuit.circuit_model_rewrites import (
    BatchMulTracerItem,
    basic_cum_expand_run,
    basic_factoring,
    batch_mul_by_tracer,
    extract_head,
    add_drop_by_match,
    batch_mul,
    multiple_batch_mul_by_tracer,
    run_factor_distribute,
)
from interp.circuit.get_update_node import (
    FalseMatcher,
    NodeMatcher,
    TrueMatcher,
    TypeMatcher,
    EqMatcher,
    AnyMatcher,
    NameMatcher as NM,
    NodeUpdater as NU,
    IterativeNodeMatcher as INM,
    FunctionIterativeNodeMatcher as F,
    BasicFilterIterativeNodeMatcher as BF,
    Rename,
    Replace,
    RegexMatcher as RE,
    replace_circuit,
    sub_name,
)
from interp.circuit.function_rewrites import one_hot_log_loss
from interp.tools.type_utils import assert_never
from interp.circuit.algebric_rewrite import (
    MulRearrangeSpec,
    MulRearrangeSpecSub,
    NamedItem,
    distribute,
    drop_mul_ones,
    einsum_remove_unsqueeze,
    equivalence_partition,
    explicit_reduce,
    factor_add_of_mul_to_mul_of_add,
    flatten_adds,
    try_flatten_adds,
    flatten_adds_matching,
    flatten_muls,
    permute_to_einsum,
    push_down_index,
    push_down_permute_via_einsum,
    rearrange_muls,
    remove_add_times_one,
    residual_rewrite,
    split_einsum_concat,
    unary_add_to_scalar_mul,
    fuse_single_einsum,
    weighted_to_unweighted_add,
    try_drop_mul_ones,
    try_fuse_einsum_rearrange,
)
from interp.circuit.circuit_simplification import basic_simp
from interp.circuit.cum_algo import (
    cumulant_function_derivative_estim,
    rewrite_cum_to_circuit_of_cum,
)
from interp.circuit.print_circuit import PrintCircuit
from interp.circuit.var import DiscreteVar, StoredCumulantVar
import interp.tools.optional as op
from interp.circuit.circuit_compiler.compiler import evaluate_circuits
from interp.circuit.computational_node import (
    Add,
    Concat,
    Einsum,
    GeneralFunction,
    Index,
    UnaryRearrange,
    log_exp_p_1_fn,
    softmax_fn,
)
from interp.circuit.circuit import Circuit, MemoizedFn
from interp.circuit.constant import ArrayConstant, FloatConstant, One, Zero
from interp.circuit.cumulant import Cumulant
from interp.circuit.sample_transform import (
    AllRecursiveNonTrivialCumulants,
    RandomSampleSpec,
    RunDiscreteVarAllSpec,
    SampleSpec,
    StoredIdxsWeightsDiscreteSample,
    center,
    PartitionedSamplingFull,
    eps_attrib,
    sample_transform_deep,
)
from interp.circuit.circuit_utils import cast_circuit
from interp.tools.indexer import TORCH_INDEXER as I, SLICER as S
from interp.tools.interpretability_tools import (
    begin_token,
    get_interp_tokenizer,
    print_max_min_by_tok_k_torch,
    single_tokenize,
    toks_to_string_list,
)

from interp.circuit.scope_rewrites import basic_factor_distribute
from interp.circuit.projects.punct.rewrites import (
    approx_head_diag_masks_a0,
    expand_probs_a1_more,
    get_a1_probs_deriv_expand,
    get_trivial_expand_embeds_and_attn,
    expand_and_factor_log_probs,
    get_log_probs_cov_expand,
)
from interp.circuit.projects.estim_helper import EstimHelper
from interp.circuit.projects.interp_utils import (
    ChildDerivInfo,
    get_items,
    add_path,
    print_for_scope,
    run_scope_estimate,
)
from interp.circuit.projects.punct.punct_interp import (
    interp_for_embeds_direct,
    interp_for_multip_a1_k3,
    interp_expected_probs_a0,
    interp_a1_expected_probs,
    interp_snd_cum_for_paths,
    interp_thrd_cum_for_paths,
    interp_thrd_cum_overall,
    interp_snd_cum_overall,
)  # TODO cleanup imports

# from interp.circuit.projects.setup_utils import
# from interp.circuit.projects.induction.utils import *
from pathlib import PurePath as PP

# %% [markdown]
# # Goal: find induction heads with the cumulants method
# Setup
# WithCompilerSettings(DO_FIX_NUMERICS=True).global_update() # TODO hopefully fine
#%%
# DATA SHIT
print("WARN: not using Ryan stuff")
data_rrfs = os.path.expanduser(f"~/rrfs/pretraining_datasets/owt_tokens_int16/0.pt")
data_suffix = "name_data/data-2022-07-30.pt"
data_local = os.path.expanduser(f"~/{data_suffix}")
try:
    data_full = torch.load(data_local)
except FileNotFoundError:
    data_full = torch.load(data_rrfs)
toks = data_full["tokens"].long() + 32768
lens = data_full["lens"].long()


def d(tokens, tokenizer=model.tokenizer):
    return tokenizer.decode(tokens)


if False:
    SEQ_LEN = 10
    print(f"WARN: SEQ_LEN = {SEQ_LEN}")
    MODEL_ID = "attention_only_four_layers_untied"
    MODEL_ID = "attention_only_two_layers_untied"
    # MODEL_ID = "jan5_attn_only_two_layers"
    DATASET_SIZE = 8000  # total data points is twice this...
    DATASET_DIR = PP("/home/arthur/rrfs/arthur/induction/data7/")
    MODIFY_DATASETS = False
    TRIM_TO_SIZE = False
    FIND_SAME_TOKEN = False

    DATASET_PATH = DATASET_DIR / "ind.pt"
    MADE_DATA = os.path.exists(DATASET_PATH)
    VOCAB_SIZE = 50259
    if os.path.exists(DATASET_PATH):
        smol = torch.load(str(DATASET_PATH))
        print("Trying to decode ...")
        print(d(smol[0, :]))
        print("... done.")
    else:
        print("Rip, no smol found")
        if not os.path.exists(DATASET_DIR):
            print(f"Made {str(DATASET_DIR)}")
            os.mkdir(DATASET_DIR)
#%%


def perplexity(losses):
    return torch.exp(torch.mean(losses))


def bpb(losses):
    """Cursed EleutherAI value"""
    return (0.29335 / np.log(2)) * losses


tot = 0


def get_loss(model, tokens):
    losses = model(
        tokens,
        return_type="loss",
    )
    return losses.mean().item()


tot = []


def get_bpbs(model_name, manual_eos=None):
    print("Loading model", model_name)
    model = EasyTransformer.from_pretrained(model_name)
    print("Done")
    tot = [[]]

    print(model.cfg)

    for idx in tqdm(range(100)):  # range(len(lens)):
        cur = torch.cat(
            (
                torch.tensor([model.tokenizer.pad_token_id])
                if manual_eos is None
                else torch.tensor([manual_eos]),
                toks[torch.sum(lens[:idx]) : torch.sum(lens[: idx + 1])],
            )
        )
        cur_tokens = cur.unsqueeze(0)[:, :1024]

        losses = get_loss(model, cur_tokens)
        tot[-1].append(losses)

        if idx > 100:
            break

    bs = [bpb(t) for t in tot[-1]]
    return torch.tensor(bs)


#%%

for model_name in [
    "gpt2",
    "EleutherAI/gpt-neo-125M",
    "gpt2-large",
    "EleutherAI/gpt-neo-1.3B",
    "gpt2-xl",
    "EleutherAI/gpt-neo-2.7B",
]:
    bs = get_bpbs(model_name, manual_eos=0)
    print(
        f"Model {model_name} bpb: {bs.mean()} +- {bs.std()}"
    )  # 1.22 and 1.04 according to the table. Checks out!
