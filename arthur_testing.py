#%% [markdown]
# Arthur investigation into dropout
from copy import deepcopy
import torch

# from easy_transformer.experiments import get_act_hook
from experiments import get_act_hook
# from induction_utils import path_patching_attribution, prepend_padding, patch_all

assert torch.cuda.device_count() == 1
from tqdm import tqdm
import pandas as pd
import torch
import torch as t
from EasyTransformer import (
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
# from ioi_dataset import (
#     IOIDataset,
# )
# from ioi_utils import (
#     path_patching,
#     max_2d,
#     CLASS_COLORS,
#     e,
#     show_pp,
#     show_attention_patterns,
#     scatter_attention_and_contribution,
# )
from random import randint as ri
# from easy_transformer.experiments import get_act_hook
# from ioi_circuit_extraction import (
#     do_circuit_extraction,
#     get_heads_circuit,
#     CIRCUIT,
# )
import random as rd
# from ioi_utils import logit_diff, probs
# from ioi_utils import get_top_tokens_and_probs as g

ipython = get_ipython()
if ipython is not None:
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")
#%% [markdown]
# Initialise model (use larger N or fewer templates for no warnings about in-template ablation)

if "from_pretrained" in dir(EasyTransformer):
    gpt2 = EasyTransformer.from_pretrained("gpt2", device="cpu")
    gpt2.set_use_attn_result(True)
else:
    gpt2 = EasyTransformer("gpt2", use_attn_result=True)

# opt = EasyTransformer.from_pretrained("facebook/opt-125m").cuda()
# opt.set_use_attn_result(True)

# neo = EasyTransformer.from_pretrained("EleutherAI/gpt-neo-125M").cuda()
# neo.set_use_attn_result(True)

# solu = EasyTransformer.from_pretrained("solu-10l-old").cuda()
# solu.set_use_attn_result(True)

# distil = EasyTransformer.from_pretrained("distilgpt2").cuda()
# distil.set_use_attn_result(True)

model = gpt2
model_names = ["gpt2", "opt", "neo", "solu"]

#%%

from experiments import EasyPatching, PatchingConfig, ExperimentMetric # easytransformer. ...
import torch.nn.functional as F

source_facts = ["Steve Jobs founded Apple", "Bill Gates founded Microsoft"]
target_facts = ["Bill Gates founded Microsoft", "Steve Jobs founded Apple"]

source_labels = [ " Apple", " Microsoft"]
target_labels = [ " Microsoft"," Apple"]

source_tokens = model.to_tokens(source_labels).squeeze() # .cuda()
target_tokens = model.to_tokens(target_labels).squeeze() # .cuda()

tokens_pos = [2,2] # The position of "founded" in the target sentences, where to get the next token prediction

def fact_transfer_score(model, target_dataset):
    # target_dataset = model.to_tokens(target_dataset).squeeze().cuda()
    logits = model(target_dataset)
    log_probs = F.log_softmax(logits, dim=-1)
    logit_diff = (log_probs[torch.arange(len(target_tokens)),tokens_pos, target_tokens] - # logit target - logit source (positive by default)
                  log_probs[torch.arange(len(source_tokens)), tokens_pos,source_tokens])
    # print(logit_diff)
    return logit_diff.mean() 

metric = ExperimentMetric(fact_transfer_score, target_facts, relative_metric=False)
config = PatchingConfig(source_dataset=source_facts, target_dataset=target_facts, target_module="attn_head", head_circuit="v",  cache_act=True, verbose=False)
patching = EasyPatching(model, config, metric)
result = patching.run_patching()
px.imshow(result, labels={'y':'Layer', 'x':'Head'}, color_continuous_scale='Blues', title="Absolute Log Logit Prob Difference After Patching, on commit 14e408b").show()