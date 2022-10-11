#%%
# % TODO: ablations last
# % and 2.2 improvements: do things with making more specific to transformers
# % ablations later
# % back reference equations
# % not HYPOTHESISE, do the computationally intractable
# % do completeness, minimality NOT methods first
#%%
from time import ctime
import io
from easy_transformer import EasyTransformer
from functools import partial
from ioi_utils import logit_diff, probs
import logging
import sys
from ioi_circuit_extraction import *
import optuna
from ioi_dataset import *
from ioi_utils import max_2d
import IPython
from tqdm import tqdm
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

from ioi_dataset import (
    IOIDataset,
    NOUNS_DICT,
    NAMES,
    gen_prompt_uniform,
    BABA_TEMPLATES,
    ABBA_TEMPLATES,
)
from ioi_utils import (
    attention_on_token,
    clear_gpu_mem,
    show_tokens,
    show_pp,
    show_attention_patterns,
    safe_del,
)

ipython = get_ipython()
if ipython is not None:
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")

def e(mess=""):
    print_gpu_mem(mess)
    torch.cuda.empty_cache()
# %%
model = EasyTransformer("gpt2", use_attn_result=True).cuda()
N = 100
ioi_dataset = IOIDataset(prompt_type="mixed", N=N, tokenizer=model.tokenizer)
abca_dataset = ioi_dataset.gen_flipped_prompts("S2")  # we flip the second b for a random c
acca_dataset = ioi_dataset.gen_flipped_prompts("S")
acba_dataset = ioi_dataset.gen_flipped_prompts("S1")
adea_dataset = ioi_dataset.gen_flipped_prompts("S").gen_flipped_prompts("S1")
all_diff_dataset = adea_dataset.gen_flipped_prompts("IO")

from ioi_utils import logit_diff
#%% [markdown] Add some ablation of MLP0 to try and tell what's up
model.reset_hooks()
metric = ExperimentMetric(metric=logit_diff, dataset=abca_dataset, relative_metric=True)
config = AblationConfig(
    abl_type="random",
    mean_dataset=abca_dataset.text_prompts,
    target_module="mlp",
    head_circuit="result",
    cache_means=True,
    verbose=False,
    nb_metric_iteration=1,
    max_seq_len=ioi_dataset.max_len,
)
abl = EasyAblation(model, config, metric) # , mean_by_groups=True, groups=ioi_dataset.groups)
e()

ablate_these = [1, 2, 3] # single numbers for MLPs, tuples for heads
# ablate_these = max_2d(-result, 10)[0] 
ablate_these += [(5, 9),
 (5, 8),
 (0, 10),
 (4, 6),
 (3, 10),
 (4, 0),
 (3, 8),
 (3, 7),
 (5, 2),
 (6, 5)]
ablate_these = []
# run some below cell to see the max impactful heads

for this in ablate_these:
    if isinstance(this, int):
        layer = this
        head_idx = None
        cur_tensor_name = f"blocks.{layer}.hook_mlp_out"
    elif isinstance(this, tuple):
        layer, head_idx = this
        cur_tensor_name = f"blocks.{layer}.attn.hook_result"
    else:
        raise ValueError(this)

    def ablation_hook(z, act, hook):  
        # batch, seq, head dim, because get_act_hook hides scary things from us
        # TODO probably change this to random ablation when that arrives
        cur_layer = int(hook.name.split(".")[1])
        cur_head_idx = hook.ctx["idx"]

        # assert hook.name == cur_tensor_name, (hook.name, cur_tensor_name)
        # sad, that only works when cur_tensor_name doesn't change (?!)
        assert len(list(z.shape)) == 3, z.shape
        assert list(z.shape) == list(act.shape), (z.shape, act.shape)

        z[torch.arange(ioi_dataset.N), ioi_dataset.word_idx["S2"]] = act[torch.arange(ioi_dataset.N), ioi_dataset.word_idx["S2"]]  # hope that we don't see changing values of mean_cached_values...
        return z

    cur_hook = get_act_hook(
        ablation_hook,
        alt_act=abl.mean_cache[cur_tensor_name],
        idx=head_idx, 
        dim=2 if head_idx is not None else None,
    )
    model.add_hook(cur_tensor_name, cur_hook)

# [markdown] After adding some hooks we see that yeah MLP0 ablations destroy S2 probs -> this one is at end
#%%
my_toks = [
    2215,
    5335,
    290,
    1757,
    1816,
    284,
    262,
    3650,
    11,
    1757,
    2921,
    257,
    4144,
    284,
    5335,
] # this is the John and Mary one

mary_tok = 5335
john_tok = 1757

model.reset_hooks()
logits = model(torch.Tensor([my_toks]).long()).detach().cpu()
assert mary_tok in torch.topk(logits[0, -2], 1).indices, (torch.topk(logits[0, -2], 5), logits[0, -2, john_tok])
# mary_res = logits[0, -2, mary_tok]
# john_res = logits[0, -2, john_tok]

def replace(my_list, a, b):
    return [b if x == a else x for x in my_list]

from random import randint as ri
from copy import deepcopy
cnt=0
bet=0
bet_sub = 0
bet_sub_2 = 0
for it in tqdm(range(1000)):
    cur_list = deepcopy(my_toks)
    new_mary_tok = my_list[ri(0, -1+len(my_list))] # ri(0, 50_000)
    new_john_tok = my_list[ri(0, -1+len(my_list))] # ri(0, 50_000)
    cur_list = replace(cur_list, mary_tok, new_mary_tok)
    cur_list = replace(cur_list, john_tok, new_john_tok)
    logits = model(torch.Tensor([cur_list]).long()).detach().cpu()[0, -2]
    top_100 = torch.topk(logits, 100).indices
        
    if logits[new_mary_tok] > logits[new_john_tok]:
        bet+=1    

    if new_mary_tok in top_100 or new_john_tok in top_100:
        cnt+=1

        if logits[new_mary_tok] > logits[new_john_tok]:
            bet_sub+=1    
        else:
            bet_sub_2+=1
#%%
NEW_CIRCUIT = {
    # old name mover
    (9, 6): ["S2", "end"],
    (9, 9): ["S+1", "end"],
    (10, 0): ["end"],
    # old s2 inhibition
    (7, 3): ["S2", "end"],
    (7, 9): ["S+1", "end"],
    (10, 7): [],
    (11, 10): [],
    # old induction
    (5, 5): ["end"],
    (5, 8): ["S"],
    (5, 9): [],
    (6, 9): [],
    # old duplicate
    (0, 1): ["IO"],
    (0, 10): ["end"],
    (3, 0): [],
    # old previous token
    (2, 2): [],
    (2, 9): ["S", "end"],
    (4, 11): ["S2"],
}

NEGS = {
    (10, 7): ["end"],
    (11, 10): ["end"],
}

model.reset_hooks()
e()
whole_circuit_base, whole_std = logit_diff(model, ioi_dataset.text_prompts, std=True)
print(f"{whole_circuit_base=} {whole_std=}")

heads_to_keep_new = {}
for head in NEW_CIRCUIT.keys():
    heads_to_keep_new[head] = get_extracted_idx(NEW_CIRCUIT[head], ioi_dataset)
e()
new_model, _ = do_circuit_extraction(
    model=model,
    heads_to_keep=heads_to_keep_new,
    mlps_to_remove={},
    ioi_dataset=ioi_dataset,
)
e()
new_circuit_base, new_std = logit_diff(new_model, ioi_dataset.text_prompts, std=True)
print(f"{new_circuit_base=} {new_std=}")
model.reset_hooks()
heads_to_keep_neg_new = heads_to_keep_new.copy()
heads_to_keep = {}
for circuit_class in CIRCUIT.keys():
    if circuit_class == "negative":
        continue
    for head in CIRCUIT[circuit_class]:
        heads_to_keep[head] = get_extracted_idx(RELEVANT_TOKENS[head], ioi_dataset)
heads_to_keep_neg = heads_to_keep.copy()
for head in NEGS.keys():
    heads_to_keep_neg_new[head] = get_extracted_idx(NEGS[head], ioi_dataset)
    heads_to_keep_neg[head] = get_extracted_idx(NEGS[head], ioi_dataset)
e()
calib_model, _ = do_circuit_extraction(
    model=model,
    heads_to_keep=heads_to_keep_neg,
    mlps_to_remove={},
    ioi_dataset=ioi_dataset,
)
e()
calib_model_base, calib_std = logit_diff(
    calib_model, ioi_dataset.text_prompts, std=True
)
print(f"{calib_model_base=} {calib_std=}")
# %%
seq_len = ioi_dataset.toks.shape[1]

for mlp in range(12):
    model.reset_hooks()
    calib_model, _ = do_circuit_extraction(
        model=model,
        heads_to_keep=heads_to_keep_neg,
        mlps_to_remove={mlp: [list(range(seq_len)) for _ in range(N)]},
        ioi_dataset=ioi_dataset,
    )
    e()
    mlp_base, mlp_std = logit_diff(calib_model, ioi_dataset.text_prompts, std=True)
    print(f"{mlp} {mlp_base=} {mlp_std=}")
#%% # quick S2 experiment
from ioi_circuit_extraction import ARTHUR_CIRCUIT

heads_to_keep = {}

for circuit_class in ARTHUR_CIRCUIT.keys():
    for head in ARTHUR_CIRCUIT[circuit_class]:
        heads_to_keep[head] = get_extracted_idx(RELEVANT_TOKENS[head], ioi_dataset)

model, abl = do_circuit_extraction(
    model=model,
    heads_to_keep=heads_to_keep,
    mlps_to_remove={},
    ioi_dataset=ioi_dataset,
)

for layer, head_idx in [(7, 9), (8, 6), (7, 3), (8, 10)]:
    # use abl.mean_cache
    cur_tensor_name = f"blocks.{layer}.attn.hook_v"
    s2_token_idxs = get_extracted_idx(["S2"], ioi_dataset)
    mean_cached_values = (
        abl.mean_cache[cur_tensor_name][:, :, head_idx, :].cpu().detach()
    )

    def s2_v_ablation_hook(
        z, act, hook
    ):  # batch, seq, head dim, because get_act_hook hides scary things from us
        cur_layer = int(hook.name.split(".")[1])
        cur_head_idx = hook.ctx["idx"]

        assert hook.name == f"blocks.{cur_layer}.attn.hook_v", hook.name
        assert len(list(z.shape)) == 3, z.shape
        assert list(z.shape) == list(act.shape), (z.shape, act.shape)

        true_s2_values = z[:, s2_token_idxs, :].clone()
        z = (
            mean_cached_values.cuda()
        )  # hope that we don't see chaning values of mean_cached_values...
        z[:, s2_token_idxs, :] = true_s2_values

        return z

    cur_hook = get_act_hook(
        s2_v_ablation_hook,
        alt_act=abl.mean_cache[cur_tensor_name],
        idx=head_idx,
        dim=2,
    )
    model.add_hook(cur_tensor_name, cur_hook)

new_ld, new_ld_std = logit_diff(model, ioi_dataset.text_prompts, std=True)
new_ld, new_ld_std
#%% # look at what's affecting the V stuff
from ioi_circuit_extraction import ARTHUR_CIRCUIT
from ioi_utils import probs
heads_to_keep = {}
for circuit_class in CIRCUIT.keys():
    for head in CIRCUIT[circuit_class]:
        # if head[0] <= 6: continue # let's think about the effect before S2 ..
        heads_to_keep[head] = get_extracted_idx(RELEVANT_TOKENS[head], ioi_dataset)

# early_heads = [(layer, head_idx) for layer in list(range(6[])) for head_idx in list(range(12))]

model, abl = do_circuit_extraction(
    model=model,
    heads_to_keep=heads_to_keep,
    mlps_to_remove={},
    ioi_dataset=ioi_dataset,
    # exclude_heads=early_heads,
)
init_probs = probs(model, ioi_dataset)
print(f"{init_probs=}")

vprobs = torch.zeros(7, 12)

for layer in range(7):
    for head_idx in range(12):
        print(layer, head_idx)
        heads_to_keep = {}
        for circuit_class in ARTHUR_CIRCUIT.keys():
            for head in ARTHUR_CIRCUIT[circuit_class]:
                if head[0] <= 6:
                    continue
                heads_to_keep[head] = get_extracted_idx(
                    RELEVANT_TOKENS[head], ioi_dataset
                    )
        new_early_heads = early_heads.copy()
        new_early_heads.remove((layer, head_idx))
        model, abl = do_circuit_extraction(
            model=model,
            heads_to_keep=heads_to_keep,
            mlps_to_remove={},
            ioi_dataset=ioi_dataset,
            exclude_heads=new_early_heads,
        )
        vprobs[layer][head_idx] = probs(model, ioi_dataset)
#%% # wait, do even the very first plots work for the IO probs metric??
vals = torch.zeros(12, 12)
from ioi_circuit_extraction import (
    ARTHUR_CIRCUIT,
    get_heads_circuit,
    CIRCUIT,
    do_circuit_extraction,
)
from ioi_utils import probs

old_probs = probs(model, ioi_dataset)

for layer in range(12):
    print(layer)
    for head in range(12):
        heads_to_keep = get_heads_circuit(ioi_dataset, excluded=[], circuit=CIRCUIT)
        torch.cuda.empty_cache()

        model.reset_hooks()
        model, _ = do_circuit_extraction(
            model=model,
            heads_to_remove={
                (layer, head): [
                    list(range(ioi_dataset.word_idx["end"][i] + 1)) for i in range(N)
                ]
            },
            mlps_to_remove={},
            ioi_dataset=ioi_dataset,
            # exclude_heads=[(layer, head)],
        )
        torch.cuda.empty_cache()
        new_probs = probs(model, ioi_dataset)
        vals[layer, head] = new_probs - old_probs

show_pp(vals)
#%%
vals2 = torch.zeros(12)
for layer in range(12):
    print(layer)

    heads_to_keep = get_heads_circuit(ioi_dataset, excluded=[], circuit=CIRCUIT)
    torch.cuda.empty_cache()

    model.reset_hooks()
    model, _ = do_circuit_extraction(
        model=model,
        mlps_to_remove={
            (layer): [list(range(ioi_dataset.word_idx["end"][i] + 1)) for i in range(N)]
        },
        heads_to_remove={},
        ioi_dataset=ioi_dataset,
    )
    torch.cuda.empty_cache()
    new_probs = probs(model, ioi_dataset)
    vals2[layer] = new_probs - old_probs
#%%
show_pp(vals2.unsqueeze(0), title="MLP removal")
#%%
# %%
show_scatter = True
circuit_perf_scatter = []
eps = 1.2

# by points
if show_scatter:
    fig = go.Figure()
    all_xs = []
    all_ys = []

    for i, circuit_class in enumerate(set(circuit_perf.removed_group)):
        xs = list(
            circuit_perf[circuit_perf["removed_group"] == circuit_class][
                "cur_metric_broken"
            ]
        )
        ys = list(
            circuit_perf[circuit_perf["removed_group"] == circuit_class][
                "cur_metric_cobble"
            ]
        )

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                # hover_data=["sentence", "template"], # TODO get this working
                mode="markers",
                marker=dict(color=CLASS_COLORS[circuit_class], size=3),
                # name=circuit_vlass,
                showlegend=False,
                # color=CLASS_COLORS[circuit_class],
                # opacity=1.0,
            )
        )

        all_xs += xs
        all_ys += ys
        plot_ellipse(
            fig,
            xs,
            ys,
            color=CLASS_COLORS[circuit_class],
            name=circuit_class,
        )

    minx = min(min(all_xs), min(all_ys))
    maxx = max(max(all_xs), max(all_ys))
    fig.update_layout(
        shapes=[
            dict(
                type="line",
                xref="x",
                x0=minx,
                x1=maxx,
                yref="y",
                y0=minx,
                y1=maxx,
            )
        ]
    )

    xs = np.linspace(minx, maxx, 100)
    ys_max = xs + eps
    ys_min = xs - eps

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys_min,
            mode="lines",
            name="THIS ONE IS HIDDEN",
            showlegend=False,
            line=dict(color="grey"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys_max,
            mode="lines",
            name=f"Completeness region, epsilon={eps}",
            fill="tonexty",
            line=dict(color="grey"),
        )
    )

    fig.update_xaxes(gridcolor="black", gridwidth=0.1)
    fig.update_yaxes(gridcolor="black", gridwidth=0.1)
    fig.update_layout(paper_bgcolor="white", plot_bgcolor="white")
    fig.write_image(f"svgs/circuit_completeness_at_{ctime()}.svg")
    fig.show()
#%%
s = """ball
bat
bed
book
boy
bun
can
cake
cap
car
cat
cow
cub
cup
dad
day
dog
doll
dust
fan
feet
girl
gun
hall
hat
hen
jar
kite
man
map
men
mom
pan
pet
pie
pig
pot
rat
son
sun
toe
tub
van
apple
arm
banana
bike
bird
book
chin
clam
class
clover
club
corn
crayon
crow
crown
crowd
crib
desk
dime
dirt
dress
fang
field
flag
flower
fog
game
heat
hill
home
horn
hose
joke
juice
kite
lake
maid
mask
mice
milk
mint
meal
meat
moon
mother
morning
name
nest
nose
pear
pen
pencil
plant
rain
river
road
rock
room
rose
seed
shape
shoe
shop
show
sink
snail
snake
snow
soda
sofa
star
step
stew
stove
straw
string
summer
swing
table
tank
team
tent
test
toes
tree
vest
water
wing
winter
woman
women
alarm
animal
aunt
bait
balloon
bath
bead
beam
bean
bedroom
boot
bread
brick
brother
camp
chicken
children
crook
deer
dock
doctor
downtown
drum
dust
eye
family
father
fight
flesh
food
frog
goose
grade
grandfather
grandmother
grape
grass
hook
horse
jail
jam
kiss
kitten
light
loaf
lock
lunch
lunchroom
meal
mother
notebook
owl
pail
parent
park
plot
rabbit
rake
robin
sack
sail
scale
sea
sister
soap
song
spark
space
spoon
spot
spy
summer
tiger
toad
town
trail
tramp
tray
trick
trip
uncle
vase
winter
water
week
wheel
wish
wool
yard
zebra
women
actor
airplane
airport
army
baseball
beef
birthday
boy
brush
bushes
butter
cast
cave
cent
cherries
cherry
cobweb
coil
cracker
dinner
eggnog
elbow
face
fireman
flavor
gate
glove
glue
goldfish
goose
grain
hair
haircut
hobbies
holiday
hot
jellyfish
ladybug
mailbox
number
oatmeal
pail
pancake
pear
pest
popcorn
queen
quicksand
quiet
quilt
rainstorm
scarecrow
scarf
stream
street
sugar
throne
toothpaste
twig
volleyball
wood
wrench
advice
anger
answer
apple
arithmetic
badge
basket
basketball
battle
beast
beetle
beggar
brain
branch
bubble
bucket
cactus
cannon
cattle
celery
cellar
cloth
coach
coast
crate
cream
daughter
donkey
drug
earthquake
feast
fifth
finger
flock
frame
furniture
geese
ghost
giraffe
governor
honey
hope
hydrant
icicle
income
island
jeans
judge
lace
lamp
lettuce
marble
month
north
ocean
patch
plane
playground
poison
riddle
rifle
scale
seashore
sheet
sidewalk
skate
slave
sleet
smoke
stage
station
thrill
throat
throne
title
toothbrush
turkey
underwear
vacation
vegetable
visitor
voyage
year
able
achieve
acoustics
action
activity
aftermath
afternoon
afterthought
apparel
appliance
beginner
believe
bomb
border
boundary
breakfast
cabbage
cable
calculator
calendar
caption
carpenter
cemetery
channel
circle
creator
creature
education
faucet
feather
friction
fruit
fuel
galley
guide
guitar
health
heart
idea
kitten
laborer
language
lawyer
linen
locket
lumber
magic
minister
mitten
money
mountain
music
partner
passenger
pickle
picture
plantation
plastic
pleasure
pocket
police
pollution
railway
recess
reward
route
scene
scent
squirrel
stranger
suit
sweater
temper
territory
texture
thread
treatment
veil
vein
volcano
wealth
weather
wilderness
wren
wrist
writer"""
s = s.split("\n")

my_list = []

for st in range(len(s)):
    thing = model.tokenizer(" "+s[st])["input_ids"]
    if len(thing) != 1: continue
    my_list.append(thing[0])
#%% # Q: can we just replace S2 Inhibition Heads with 1.0 attention to S2?
# A: pretty much yes

from ioi_circuit_extraction import CIRCUIT
from ioi_utils import probs, logit_diff
circuit = CIRCUIT.copy()

heads_to_keep = get_heads_circuit(
    ioi_dataset,
    circuit=circuit,
)

heads = circuit["s2 inhibition"].copy()

for change in [False, True]:
    model.reset_hooks()
    
    model, abl = do_circuit_extraction(
        model=model,
        heads_to_keep=heads_to_keep,
        mlps_to_remove={},
        ioi_dataset=ioi_dataset,
    )

    if change:
        for layer, head in heads:
            hook_name = f"blocks.{layer}.attn.hook_attn"
            def s2_ablation_hook(z, act, hook):  # batch, seq, head dim, because get_act_hook hides scary things from us
                assert z.shape == act.shape, (z.shape, act.shape)
                z = act
                return z

            act = torch.zeros(size=(ioi_dataset.N, model.cfg.n_heads, ioi_dataset.max_len, ioi_dataset.max_len))
            act[torch.arange(ioi_dataset.N), :, ioi_dataset.word_idx["end"][:ioi_dataset.N], ioi_dataset.word_idx["S2"][:ioi_dataset.N]] = 0.5
            cur_hook = get_act_hook(
                s2_ablation_hook,
                alt_act=act,
                idx=head,
                dim=1,
            )
            model.add_hook(hook_name, cur_hook)

    io_probs = probs(model, ioi_dataset)
    print(f" {logit_diff(model, ioi_dataset)}, {io_probs=}") 
#%% evidence for the S2 story
# ablating V for everywhere except S2 barely affects LD. But ablating all V has LD go to almost 0

model.reset_hooks()
model, abl = do_circuit_extraction(
    model=model,
    heads_to_keep=heads_to_keep,
    mlps_to_remove={},
    ioi_dataset=ioi_dataset,
)

for layer, head_idx in [(7, 9), (8, 6), (7, 3), (8, 10)]:
    # break
    cur_tensor_name = f"blocks.{layer}.attn.hook_q"
    s2_token_idxs = get_extracted_idx(["S2"], ioi_dataset)
    mean_cached_values = abl.mean_cache[cur_tensor_name][:, :, head_idx, :].cpu().detach()

    def s2_v_ablation_hook(z, act, hook):  # batch, seq, head dim, because get_act_hook hides scary things from us
        cur_layer = int(hook.name.split(".")[1])
        cur_head_idx = hook.ctx["idx"]

        assert hook.name == f"blocks.{cur_layer}.attn.hook_q", hook.name
        assert len(list(z.shape)) == 3, z.shape
        assert list(z.shape) == list(act.shape), (z.shape, act.shape)

        z = mean_cached_values.cuda()  # hope that we don't see chaning values of mean_cached_values...
        return z

    cur_hook = get_act_hook(
        s2_v_ablation_hook,
        alt_act=abl.mean_cache[cur_tensor_name],
        idx=head_idx,
        dim=2,
    )
    model.add_hook(cur_tensor_name, cur_hook)

new_ld = logit_diff(model, ioi_dataset)
new_probs = probs(model, ioi_dataset)
print(f"{new_ld=}, {new_probs=}")
#%% # new shit: attention probs on S2 is the score
heads_to_patch = circuit["s2 inhibition"].copy()
attn_circuit_template = "blocks.{patch_layer}.attn.hook_v"
cache_names = set([attn_circuit_template.format(patch_layer=patch_layer) for patch_layer, _ in heads_to_patch])

logit_diffs = torch.zeros(size=(12, 12))
mlps = torch.zeros(size=(12,))
model.reset_hooks()

S2_LAYER = 7
S2_HEAD = 9
metric = partial(attention_on_token, head_idx=S2_HEAD, layer=S2_LAYER, token="S2")
metric = partial(attention_on_token, head_idx=9, layer=9, token="IO")

model.reset_hooks()
base_metric = metric(model, ioi_dataset)
print(f"{base_metric=}")

experiment_metric = ExperimentMetric(metric=metric, dataset=ioi_dataset, relative_metric=False)
config = AblationConfig(
    abl_type="random",
    mean_dataset=abca_dataset.text_prompts,
    target_module="attn_head",
    head_circuit="result",
    cache_means=True,
    verbose=False,
    nb_metric_iteration=1,
    max_seq_len=ioi_dataset.max_len,
)
abl = EasyAblation(model, config, experiment_metric) # , mean_by_groups=True, groups=ioi_dataset.groups)
e()
result = abl.run_experiment()
show_pp((result-base_metric).T, xlabel="head", ylabel="layer", title="Attention on S2 change when ablated")
#%%
for layer in range(12):
    for head_idx in [None] + list(range(12)):
        # do a run where we ablate (head, layer)

        if head_idx is None:
            cur_tensor_name = f"blocks.{layer}.hook_mlp_out"
            mean_cached_values = abl.mean_cache[cur_tensor_name].cpu().detach()

        else:
            cur_tensor_name = f"blocks.{layer}.attn.hook_result"
            mean_cached_values = abl.mean_cache[cur_tensor_name][:, :, head_idx, :].cpu().detach()

        def ablation_hook(z, act, hook): 
            # batch, seq, head dim, because get_act_hook hides scary things from us
            cur_layer = int(hook.name.split(".")[1])
            cur_head_idx = hook.ctx["idx"]

            assert hook.name == cur_tensor_name, hook.name
            assert len(list(z.shape)) == 3, z.shape
            assert list(z.shape) == list(act.shape), (z.shape, act.shape)

            z = mean_cached_values.cuda()  # hope that we don't see changing values of mean_cached_values...
            return z

        cur_hook = get_act_hook(
            ablation_hook,
            alt_act=abl.mean_cache[cur_tensor_name],
            idx=head_idx, # nice deals with 
            dim=2 if head_idx is not None else None, # None for MLPs
        )
        model.reset_hooks()
        model.add_hook(cur_tensor_name, cur_hook)
        cache = {}

        model.cache_some(cache, lambda x: x in cache_names)
        torch.cuda.empty_cache()
        metric(model, ioi_dataset)
        # all_cached[(layer, head_idx)] = cache[f"blocks.{S2_HEAD}.attn.hook_q"].cpu().detach()

        model.reset_hooks()
        for patch_layer, patch_head in heads_to_patch:
            def patch_in_q(z, act, hook):
                # assert hook.name == f"blocks.{patch_head}.attn.hook_q", hook.name # OOH ERR, is commenting this out ok???
                assert len(list(z.shape)) == 3, z.shape
                assert list(z.shape) == list(act.shape), (z.shape, act.shape)
                z = act.cuda()
                return z

            s2_hook = get_act_hook(
                patch_in_q,
                alt_act=cache[attn_circuit_template.format(patch_layer=patch_layer)],
                idx=patch_head, 
                dim=2,
            )
            model.add_hook(attn_circuit_template.format(patch_layer=patch_layer), s2_hook)

        ld = metric(model, ioi_dataset)

        if head_idx is None:
            mlps[layer] = ld.detach().cpu()
        else:
            logit_diffs[layer, head_idx] = ld.detach().cpu()

        print(f"{layer=}, {head_idx=}, {ld=}")

att_heads_mean_diff = logit_diffs - base_metric
show_pp(att_heads_mean_diff.T, ylabel="layer", xlabel="head", title=f"Change in logit diff: {torch.sum(att_heads_mean_diff)}")
mlps_mean_diff = mlps - base_ld
show_pp(mlps_mean_diff.T.unsqueeze(0), ylabel="layer", xlabel="head", title=f"Change in logit diff, MLPs: {torch.sum(mlps_mean_diff)}")
#%% # new experiment idea: the duplicators and induction heads shouldn't care where their attention is going, provided that
# it goes to either S or S+1.

for j in range(2, 5):
    # [batch, head_index, query_pos, key_pos] # so pass dim=1 to ignore the head
    def attention_pattern_modifier(z, hook):
        cur_layer = int(hook.name.split(".")[1])
        cur_head_idx = hook.ctx["idx"]

        assert hook.name == f"blocks.{cur_layer}.attn.hook_attn", hook.name
        assert len(list(z.shape)) == 3, z.shape  
        # batch, seq (attending_query), attending_key

        # cur = z[torch.arange(ioi_dataset.N), s2_positions, s_positions+1]
        # print(cur)
        # print(f"{cur.shape=}")
        # some_atts = torch.argmax(cur, dim=1) 
        # for i in range(20):
            # print(i, model.tokenizer.decode(ioi_dataset.toks[i][some_atts[i]]), ":", model.tokenizer.decode(ioi_dataset.toks[i][:6]))
        # print(some_atts.shape)

        # prior_stuff = []
        # for i in range(0, 2):
        #     prior_stuff.append(z[torch.arange(ioi_dataset.N), s2_positions, s_positions + i].clone())
        # for i in range(0, 2):
        #     z[torch.arange(ioi_dataset.N), s2_positions, s_positions + i] =  prior_stuff[(i + j) % 2] # +1 is the do nothing one # ([0, 1][(i+j)%2]) is way beyond scope


        # z[torch.arange(ioi_dataset.N), s2_positions, 0] = prior_stuff[(0 + j) % 2]
        # z[torch.arange(ioi_dataset.N), s2_positions, s_positions] = prior_stuff[(1 + j) % 2]

        z[torch.arange(ioi_dataset.N), s2_positions, :] = 0

        for key in ioi_dataset.word_idx.keys():
            z[torch.arange(ioi_dataset.N), s2_positions, ioi_dataset.word_idx[key]] = average_attention[(cur_layer, cur_head_idx)][key]

        return z

    F = logit_diff # or logit diff
    model.reset_hooks()
    ld = F(model, ioi_dataset)

    circuit_classes = ["s2 inhibition"]

    for circuit_class in circuit_classes:
        for layer, head_idx in circuit[circuit_class]:
            cur_hook = get_act_hook(
                attention_pattern_modifier,
                alt_act=None,
                idx=head_idx,
                dim=1,
            )
            model.add_hook(f"blocks.{layer}.attn.hook_attn", cur_hook)

    ld2 = F(model, ioi_dataset)
    print(
        f"Initially there's a logit difference of {ld}, and after permuting by {j-1}, the new logit difference is {ld2=}"
    )
#%%
heads_to_patch = circuit["duplicate token"] + circuit["induction"] # circuit["s2 inhibition"]
layers = list(set([layer for layer, head_idx in heads_to_patch]))
hook_names = [f"blocks.{l}.attn.hook_result" for l in layers]
model.reset_hooks()
cache_s2 = {}
model.cache_some(cache_s2, lambda x: x in hook_names)
logits = model(adea_dataset.text_prompts).detach()

def patch_positions(z, source_act, hook, positions=["end"]):
    for pos in positions:
        z[torch.arange(ioi_dataset.N), ioi_dataset.word_idx[pos]] = source_act[
            torch.arange(ioi_dataset.N), ioi_dataset.word_idx[pos]
        ]
    return z
patch_last_tokens = partial(patch_positions, positions=["end"])
patch_s2_token = partial(patch_positions, positions=["S2"])

for layer, head_idx in heads_to_patch:
    cur_hook = get_act_hook(
        patch_s2_token,
        alt_act=cache_s2["blocks.{}.attn.hook_result".format(layer)],
        idx=head_idx,
        dim=2,
    )
    model.add_hook(f"blocks.{layer}.attn.hook_result", cur_hook)

l = logit_diff(model, ioi_dataset)
print(f"{l=}")
model.reset_hooks()

ys = []
fig = go.Figure()
average_attention = {}
for heads_raw in circuit["s2 inhibition"]: # heads_to_patch: # [(9, 9), (9, 6), (10, 0)]:
    heads = [heads_raw]
    average_attention[heads_raw] = {}
    for idx, dataset in enumerate([ioi_dataset, abca_dataset][:1]):
        print(idx)
        cur_ys = []
        cur_stds = []
        att = torch.zeros(size=(dataset.N, dataset.max_len, dataset.max_len))
        for head in tqdm(heads):
            att += show_attention_patterns(model, [head], dataset, return_mtx=True, mode="attn") 
        att /= len(heads)
        for key in ioi_dataset.word_idx.keys():
            end_to_s2 = att[torch.arange(dataset.N), ioi_dataset.word_idx["end"][:dataset.N], ioi_dataset.word_idx[key][:dataset.N]]
            # ABCA dataset calculates S2 in trash way... so we use the IOI dataset indices
            cur_ys.append(end_to_s2.mean().item())
            cur_stds.append(end_to_s2.std().item())
            average_attention[heads_raw][key] = end_to_s2.mean().item()
        fig.add_trace(go.Bar(x=list(ioi_dataset.word_idx.keys()), y=cur_ys, error_y=dict(type="data", array=cur_stds), name=str(heads_raw))) # ["IOI", "ABCA"][idx]))

    fig.update_layout(title_text="Attention from END to S2")
    fig.show()
#%%
heads_to_measure = [(9, 6), (9, 9), (10, 0)]  # name movers
heads_by_layer = {9: [6, 9], 10: [0]}
warnings.warn("Testing the only 9.9")
layers = [9, 10]
hook_names = [f"blocks.{l}.attn.hook_attn" for l in layers]

# warnings.warn("Actually doing S Inhib stuff")    
# heads_to_measure = circuit["s2 inhibition"]
# layers = [7, 8]
# heads_by_layer = {7: [3, 9], 8: [6, 10]}
# hook_names = [f"blocks.{l}.attn.hook_attn" for l in layers]

model.reset_hooks()
cache_baseline = {}
model.cache_some(cache_baseline, lambda x: x in hook_names)  # we only cache the activation we're interested
logits = model(ioi_dataset.text_prompts).detach()

def attention_probs(
    model, text_prompts, variation=True, scale=True
):  # we have to redefine logit differences to use the new abba dataset
    """Difference between the IO and the S logits at the "to" token"""
    cache_patched = {}
    model.cache_some(cache_patched, lambda x: x in hook_names)  # we only cache the activation we're interested
    logits = model(text_prompts).detach()
    # we want to measure Mean(Patched/baseline) and not Mean(Patched)/Mean(baseline)
    # ... but do this elsewhere as otherwise we bad
    # attn score of head HEAD at token "to" (end) to token IO
    assert variation or not scale
    attn_probs_variation_by_keys = []
    for key in ["IO", "S", "S2"]:
        attn_probs_variation = []
        for i, hook_name in enumerate(hook_names):
            layer = layers[i]
            for head in heads_by_layer[layer]:
                attn_probs_patched = cache_patched[hook_name][
                    torch.arange(len(text_prompts)),
                    head,
                    ioi_dataset.word_idx["end"],
                    ioi_dataset.word_idx[key],
                ]
                attn_probs_base = cache_baseline[hook_name][
                    torch.arange(len(text_prompts)),
                    head,
                    ioi_dataset.word_idx["end"],
                    ioi_dataset.word_idx[key],
                ]
                if variation:
                    res = attn_probs_patched - attn_probs_base
                    if scale:
                        res /= attn_probs_base
                    res = res.mean().unsqueeze(dim=0)
                    attn_probs_variation.append(res)
                else:
                    attn_probs_variation.append(attn_probs_patched.mean().unsqueeze(dim=0))
        attn_probs_variation_by_keys.append(torch.cat(attn_probs_variation).mean(dim=0, keepdim=True))

    attn_probs_variation_by_keys = torch.cat(attn_probs_variation_by_keys, dim=0)
    return attn_probs_variation_by_keys.detach().cpu()
#%% Run this after having computed attention patterns for S2 on different dists. Then Induction, Duplicate Token don't affect performance *through logit diff
def attention_pattern_modifier(z, hook):
    cur_layer = int(hook.name.split(".")[1])
    cur_head_idx = hook.ctx["idx"]

    assert hook.name == f"blocks.{cur_layer}.attn.hook_attn", hook.name
    assert len(list(z.shape)) == 3, z.shape  
    # batch, seq (attending_query), attending_key

    z[torch.arange(ioi_dataset.N), ioi_dataset.word_idx["end"], :] = 0

    for key in ioi_dataset.word_idx.keys():
        z[torch.arange(ioi_dataset.N), ioi_dataset.word_idx["end"], ioi_dataset.word_idx[key]] = average_attention[(cur_layer, cur_head_idx)][key]

    return z

F = logit_diff # or logit diff
model.reset_hooks()
ld = F(model, ioi_dataset)
circuit_classes = ["s2 inhibition"]
add_these_hooks = []

for circuit_class in circuit_classes:
    for layer, head_idx in circuit[circuit_class]:
        cur_hook = get_act_hook(
            attention_pattern_modifier,
            alt_act=None,
            idx=head_idx,
            dim=1,
        )
        add_these_hooks.append((f"blocks.{layer}.attn.hook_attn", cur_hook))
        model.add_hook(*add_these_hooks[-1])

for layer, head_idx in circuit["induction"]: #  heads_to_patch:
    cur_hook = get_act_hook(
        patch_s2_token,
        alt_act=cache_s2["blocks.{}.attn.hook_result".format(layer)],
        idx=head_idx,
        dim=2,
    )
    model.add_hook(f"blocks.{layer}.attn.hook_result", cur_hook)

l = logit_diff(model, ioi_dataset)
print(f"{l=}")

model.reset_hooks()
print(f"Logit diff = {logit_diff(model, ioi_dataset)}")
# %% [markdown] Q: is it possible to patch in ACCA sentences to make things work? A: Yes!
def patch_positions(z, source_act, hook, positions=["END"]):  # we patch at the "to" token
    for pos in positions:
        z[torch.arange(ioi_dataset.N), ioi_dataset.word_idx[pos]] = source_act[
            torch.arange(ioi_dataset.N), ioi_dataset.word_idx[pos]
        ]
    return z

patch_last_tokens = partial(patch_positions, positions=["end"])
#%%  actually answered here...
config = PatchingConfig(
    source_dataset=acba_dataset.text_prompts,
    target_dataset=ioi_dataset.text_prompts,
    target_module="attn_head",
    head_circuit="result",  # we patch "result", the result of the attention head
    cache_act=True,
    verbose=False,
    patch_fn=patch_last_tokens,
    layers=(0, max(layers) - 1),
)  # we stop at layer "LAYER" because it's useless to patch after layer 9 if what we measure is attention of a head at layer 9.
metric = ExperimentMetric(partial(attention_probs, scale=True), config.target_dataset, relative_metric=False, scalar_metric=False)
patching = EasyPatching(model, config, metric)
# add_these_hooks = [] # actually, let's not add the fixed S2 Inhib thing
patching.other_hooks = add_these_hooks
result = patching.run_patching()

for i, key in enumerate(["IO", "S", "S2"]):
    fig = px.imshow(
        result[:, :, i],
        labels={"y": "Layer", "x": "Head"},
        title=f'Variation in attention probs of Head {str(heads_to_measure)} from token "to" to {key} after Patching ABC->ABB on "to"',
        color_continuous_midpoint=0,
        color_continuous_scale="RdBu",
    )
    fig.write_image(f"svgs/variation_average_nm_attn_prob_key_{key}_patching_ABC_END.svg")
    fig.show()
#%% [markdown] 
# This was: okay, so is ACCA identical for induction heads??? A: yes, and for dupes too
# Now is: Try the ACBA dataset and see what happens
e()
relevant_heads = {}

for head in circuit["duplicate token"] + circuit["induction"]:
    relevant_heads[head] = "S2"
# for head in circuit["s2 inhibition"]:
#     relevant_heads[head] = "end"
# for head in circuit["previous token"]:
    # relevant_heads[head] = "S+1"

circuit = deepcopy(CIRCUIT)
relevant_hook_names = set([f"blocks.{layer}.attn.hook_result" for layer, _ in relevant_heads.keys()])

if "alt_cache" not in dir():
    alt_cache = {}
    model.reset_hooks()
    model.cache_some(alt_cache, names=lambda name: name in relevant_hook_names)
    logits = model(acca_dataset.text_prompts)
    del logits
    e()

print("IO S S2")
for mode in ["new", "model", "circuit"]:
    model.reset_hooks()
    e()
    if mode != "model":
        new_heads_to_keep = get_heads_circuit(ioi_dataset, circuit=circuit)
        e("MiD")
        # for head in circuit["s2 inhibition"]:
        #     new_heads_to_keep.pop(head)
        model, _ = do_circuit_extraction(
            model=model,
            heads_to_keep=new_heads_to_keep,
            mlps_to_remove={},
            ioi_dataset=ioi_dataset,
            mean_dataset=abca_dataset,
        )
        e()

    if mode == "new":
        add_these_hooks = []
        for layer, head_idx in relevant_heads:
            e("inLOos")
            cur_hook = get_act_hook(
                partial(patch_positions, positions=[relevant_heads[(layer, head_idx)]]),
                alt_act=alt_cache[f"blocks.{layer}.attn.hook_result"],
                idx=head_idx,
                dim=2,
            )
            add_these_hooks.append((f"blocks.{layer}.attn.hook_result", cur_hook))
            model.add_hook(*add_these_hooks[-1])

    att_probs = attention_probs(model, ioi_dataset.text_prompts, variation=False, scale=False)
    print(f"mode {mode}:", att_probs)

    cur_logit_diff = logit_diff(model, ioi_dataset)
    cur_io_probs = probs(model, ioi_dataset)
    e("en")
    print(f"{mode=} {cur_logit_diff=} {cur_io_probs=}")

    safe_del("new_heads_to_keep")
    safe_del("add_these_hooks")
    safe_del("att_probs")
    safe_del("_")
    e()
#%%
# some [logit difference, IO probs] for the different modes
model_results = {"x":3.5492, "y":0.4955, "name":"Model"}
circuit_results = {"x":3.3414, "y":0.2854, "name":"Circuit", "textposition":"top left"}
hooked_induction_dupe_inhibition_results = {"x":2.8315, "y":0.2901, "name":"Induction, Duplication, Inhibition Heads hooked on ACC", "textposition":"bottom left"}
hooked_induction_dupe_results = {"x":3.3488, "y":0.2779, "name":"Induction and Duplication Heads hooked on ACC", "textposition":"bottom right"}
previous_token_results = {"x":1.2908, "y":0.1528, "name":"Previous Token Heads hooked on ACC"}
abc_duplicate_induction_results = {"x":0.0675, "y":0.0708, "name":"Duplicate and Induction Heads hooked on ABC (S2)", "textposition":"bottom right"}
acb_duplicate_induction_results = {"x":0.1685, "y":0.0727, "name":"Duplicate and Induction Heads hooked on ABC (S1)", "textposition":"top right"}

xs = []
ys = []
names = []
textpositions = []

for var in dir():
    if var.endswith("_results"):
        xs.append(globals()[var]["x"])
        ys.append(globals()[var]["y"])
        names.append(globals()[var]["name"])
        textpositions.append(globals()[var].get("textposition", "top center"))

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=xs,
        y=ys,
        mode="markers+text",
        text=names,
        textposition=textpositions,
        # textposition=[["top left", "bottom left"][i%2] for i in range(len(xs))],
    )
)

# set x scaling
fig.update_xaxes(range=[0, 5])

fig.update_layout(
    title="Effects of various patching experiments on circuit behaviour",
    xaxis_title="Logit Difference",
    yaxis_title="IO Probability",
)
fig.write_image(f"svgs/new_signal_plots_at_{ctime()}.svg")
fig.show()

# writing direction for S1 was here... now RIP
#%%
#%% [markdown] 
# Wait, do Induction Heads maybe depend on Duplicate Token Heads? Let's try that
# 1) patch duplicate token heads 
# 2) save the induction head outputs
# 3) patch in induction head outputs from this dist

# 1)
e()
duplicate_heads = {}
induction_heads = {}
for head in circuit["duplicate token"]:
    duplicate_heads[head] = "S2"
for head in circuit["induction"]:
    induction_heads[head] = "S2"
duplicate_hook_names = set([f"blocks.{layer}.attn.hook_result" for layer, _ in duplicate_heads.keys()])
induction_hook_names = set([f"blocks.{layer}.attn.hook_result" for layer, _ in induction_heads.keys()])
duplicate_cache = {}
model.reset_hooks()
model.cache_some(duplicate_cache, names=lambda name: name in duplicate_hook_names)
logits = model(abca_dataset.text_prompts)
del logits
e()

# 2)
model.reset_hooks()
for layer, head_idx in duplicate_heads.keys():
    cur_hook = get_act_hook(
        partial(patch_positions, positions=[duplicate_heads[(layer, head_idx)]]),
        alt_act=duplicate_cache[f"blocks.{layer}.attn.hook_result"],
        idx=head_idx,
        dim=2,
    )
    model.add_hook(f"blocks.{layer}.attn.hook_result", cur_hook)
induction_cache = {}
model.cache_some(induction_cache, names=lambda name: name in induction_hook_names)
logits = model(ioi_dataset.text_prompts)
del logits
e()

# 3)
model.reset_hooks()
for layer, head_idx in induction_heads.keys():
    cur_hook = get_act_hook(
        partial(patch_positions, positions=[induction_heads[(layer, head_idx)]]),
        alt_act=induction_cache[f"blocks.{layer}.attn.hook_result"],
        idx=head_idx,
        dim=2,
    )
    model.add_hook(f"blocks.{layer}.attn.hook_result", cur_hook)

# model.reset_hooks()
att_probs = attention_probs(model, ioi_dataset.text_prompts, variation=False, scale=False)
print(f"{att_probs=}")

cur_logit_diff = logit_diff(model, ioi_dataset)
cur_io_probs = probs(model, ioi_dataset)
e("en")
print(f"{cur_logit_diff=} {cur_io_probs=}")

safe_del("new_heads_to_keep")
safe_del("add_these_hooks")
safe_del("att_probs")
safe_del("_")
e()