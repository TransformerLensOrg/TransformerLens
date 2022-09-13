#%% [markdown]
# # Intro
# This notebook is an implementation of the IOI experiments (some with adjustments from the [presentation](https://docs.google.com/presentation/d/19H__CYCBL5F3M-UaBB-685J-AuJZNsXqIXZR-O4j9J8/edit#slide=id.g14659e4d87a_0_290)).
# It should be able to be run as by just git cloning this repo (+ some easy installs).
# Reminder of the circuit:
#
# 
# ![image_here](https://i.imgur.com/PPtTQRh.png)
# %%
import os
try: # for Arthur
    os.chdir("/home/ubuntu/my_env/lib/python3.9/site-packages/easy_transformer")
except:
    pass

from tqdm import tqdm
import pandas as pd
from interp.circuit.projects.ioi.ioi_methods import ablate_layers, get_logit_diff
import torch
import torch as t
from easy_transformer.utils import gelu_new, to_numpy, get_corner, print_gpu_mem  # helper functions
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
ipython = get_ipython()
if ipython is not None:
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")

NAMES = [
    "Michael",
    "Christopher",
    "Jessica",
    "Matthew",
    "Ashley",
    "Jennifer",
    "Joshua",
    "Amanda",
    "Daniel",
    "David",
    "James",
    "Robert",
    "John",
    "Joseph",
    "Andrew",
    "Ryan",
    "Brandon",
    "Jason",
    "Justin",
    "Sarah",
    "William",
    "Jonathan",
    "Stephanie",
    "Brian",
    "Nicole",
    "Nicholas",
    "Anthony",
    "Heather",
    "Eric",
    "Elizabeth",
    "Adam",
    "Megan",
    "Melissa",
    "Kevin",
    "Steven",
    "Thomas",
    "Timothy",
    "Christina",
    "Kyle",
    "Rachel",
    "Laura",
    "Lauren",
    "Amber",
    "Brittany",
    "Danielle",
    "Richard",
    "Kimberly",
    "Jeffrey",
    "Amy",
    "Crystal",
    "Michelle",
    "Tiffany",
    "Jeremy",
    "Benjamin",
    "Mark",
    "Emily",
    "Aaron",
    "Charles",
    "Rebecca",
    "Jacob",
    "Stephen",
    "Patrick",
    "Sean",
    "Erin",
    "Jamie",
    "Kelly",
    "Samantha",
    "Nathan",
    "Sara",
    "Dustin",
    "Paul",
    "Angela",
    "Tyler",
    "Scott",
    "Katherine",
    "Andrea",
    "Gregory",
    "Erica",
    "Mary",
    "Travis",
    "Lisa",
    "Kenneth",
    "Bryan",
    "Lindsey",
    "Kristen",
    "Jose",
    "Alexander",
    "Jesse",
    "Katie",
    "Lindsay",
    "Shannon",
    "Vanessa",
    "Courtney",
    "Christine",
    "Alicia",
    "Cody",
    "Allison",
    "Bradley",
    "Samuel",
]

		
BABA_TEMPLATES = [	
    "Then, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",	
    "Then, [B] and [A] had a lot of fun at the [PLACE]. [B] gave a [OBJECT] to [A]",	
    "Then, [B] and [A] were working at the [PLACE]. [B] decided to give a [OBJECT] to [A]",	
    "Then, [B] and [A] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to [A]",	
    "Then, [B] and [A] had a long argument, and afterwards [B] said to [A]",	
    "After [B] and [A] went to the [PLACE], [B] gave a [OBJECT] to [A]",	
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give it to [A]",	
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give the [OBJECT] to [A]",	
    "While [B] and [A] were working at the [PLACE], [B] gave a [OBJECT] to [A]",	
    "While [B] and [A] were commuting to the [PLACE], [B] gave a [OBJECT] to [A]",	
    "After the lunch, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",	
    "Afterwards, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",	
    "Then, [B] and [A] had a long argument. Afterwards [B] said to [A]",	
    "The [PLACE] [B] and [A] went to had a [OBJECT]. [B] gave it to [A]",	
    "Friends [B] and [A] found a [OBJECT] at the [PLACE]. [B] gave it to [A]",
]

ABBA_TEMPLATES = BABA_TEMPLATES[:]	

for i in range(len(ABBA_TEMPLATES)):	
    for j in range(1, len(ABBA_TEMPLATES[i])-1):	
        if ABBA_TEMPLATES[i][j-1:j+1] == "[B]":	
            ABBA_TEMPLATES[i] = ABBA_TEMPLATES[i][:j] + "A" + ABBA_TEMPLATES[i][j + 1 :]	
        elif ABBA_TEMPLATES[i][j-1:j+1] == "[A]":	
            ABBA_TEMPLATES[i] = ABBA_TEMPLATES[i][:j] + "B" + ABBA_TEMPLATES[i][j + 1 :]

VERBS = [" tried", " said", " decided", " wanted", " gave"]
PLACES = [
    "store",
    "garden",
    "restaurant",
    "school",
    "hospital",
    "office",
    "house",
    "station",
]
OBJECTS = ["ring", "kiss", "bone", "basketball", "computer", "necklace", "drink", "snack"]

ANIMALS = [
    "dog",
    "cat",
    "snake",
    "elephant",
    "beetle",
    "hippo",
    "giraffe",
    "tiger",
    "husky",
    "lion",
    "panther",
    "whale",
    "dolphin",
    "beaver",
    "rabbit",
    "fox",
    "lamb",
    "ferret",
]


def multiple_replace(dict, text):
    # from: https://stackoverflow.com/questions/15175142/how-can-i-do-multiple-substitutions-using-regex
    # Create a regular expression from the dictionary keys
    regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))

    # For each match, look-up corresponding value in dictionary
    return regex.sub(lambda mo: dict[mo.string[mo.start() : mo.end()]], text)


def iter_sample_fast(iterable, samplesize):
    results = []
    # Fill in the first samplesize elements:
    try:
        for _ in range(samplesize):
            results.append(next(iterable))
    except StopIteration:
        raise ValueError("Sample larger than population.")
    random.shuffle(results)  # Randomize their positions

    return results

NOUNS_DICT = NOUNS_DICT = {"[PLACE]": PLACES, "[OBJECT]": OBJECTS}

def gen_prompt_uniform(templates, names, nouns_dict, N, symmetric, prefixes=None):
    nb_gen = 0
    ioi_prompts = []
    while nb_gen < N:
        temp = rd.choice(templates)
        temp_id = templates.index(temp)
        name_1 = ""
        name_2 = ""
        while name_1==name_2:
            name_1 = rd.choice(names)
            name_2 = rd.choice(names)

        nouns = {}
        for k in nouns_dict:
            nouns[k] = rd.choice(nouns_dict[k])
        prompt = temp
        for k in nouns_dict:
            prompt = prompt.replace(k, nouns[k])

        if prefixes is not None:
            L = rd.randint(30, 40)
            pref = ".".join(rd.choice(prefixes).split(".")[:L])
            pref += "<|endoftext|>"
        else:
            pref = ""

        prompt1 = prompt.replace("[A]", name_1)
        prompt1 = prompt1.replace("[B]", name_2)
        prompt1 = pref+prompt1
        ioi_prompts.append({"text":prompt1, "IO": name_1, "S":  name_2, "T_ID": temp_id})
        nb_gen+=1

        if symmetric and nb_gen<N:
            prompt2 = prompt.replace("[A]", name_2)
            prompt2 = prompt2.replace("[B]", name_1)
            prompt2 = pref+prompt2
            ioi_prompts.append({"text":prompt2, "IO": name_2, "S":  name_1, "T_ID": temp_id})
            nb_gen+=1
    return ioi_prompts


def gen_flipped_prompts(prompts, names, flip=("S2", "IO")):
    """_summary_

    Args:
        prompts (List[D]): _description_
        flip (tuple, optional): First element is the string to be replaced, Second is what to replace with. Defaults to ("S2", "IO").

    Returns:
        _type_: _description_
    """
    flipped_prompts = []

    for prompt in prompts:
        t = prompt["text"].split(" ")
        prompt = prompt.copy()
        if flip[0] == "S2":
            if flip[1] == "IO":
                t[len(t) - t[::-1].index(prompt["S"]) - 1] = prompt["IO"]
            elif flip[1] == "RAND":
                rand_name = names[np.random.randint(len(names))]
                while rand_name == prompt["IO"] or rand_name == prompt["S"]:
                    rand_name = names[np.random.randint(len(names))]
                t[len(t) - t[::-1].index(prompt["S"]) - 1] = rand_name

        if flip[0] == "IO":
            if flip[1] == "RAND":
                rand_name = names[np.random.randint(len(names))]
                while rand_name == prompt["IO"] or rand_name == prompt["S"]:
                    rand_name = names[np.random.randint(len(names))]

                t[t.index(prompt["IO"])] = rand_name
                t[t.index(prompt["IO"])] = rand_name
                prompt["IO"] = rand_name
            elif flip[1] == "ANIMAL":
                rand_animal = ANIMALS[np.random.randint(len(ANIMALS))]
                t[t.index(prompt["IO"])] = rand_animal
                prompt["IO"] = rand_animal
                #print(t)
        if flip[0] == "S":
            if flip[1] == "ANIMAL":
                new_s = ANIMALS[np.random.randint(len(ANIMALS))]
            if flip[1] == "RAND":
                new_s = names[np.random.randint(len(names))]
            t[len(t) - t[::-1].index(prompt["S"]) - 1] = new_s
            t[t.index(prompt["S"])] = new_s
            prompt["S"] = new_s
        if flip[0] == "END":
            if flip[1] == "S":
                t[len(t) - t[::-1].index(prompt["IO"]) - 1] = prompt["S"]
        if flip[0] == "PUNC":
            n = []

            # separate the punctuation from the words
            for i, word in enumerate(t):
                if "." in word:
                    n.append(word[:-1])
                    n.append(".")
                elif "," in word:
                    n.append(word[:-1])
                    n.append(",")
                else:
                    n.append(word)

            # remove punctuation, important that you check for period first
            if flip[1] == "NONE":
                if "." in n:
                    n[n.index(".")] = ""
                elif "," in n:
                    n[len(n) - n[::-1].index(",") - 1] = ""

            # remove empty strings
            while "" in n:
                n.remove("")

            # add punctuation back to the word before it
            while "," in n:
                n[n.index(",") - 1] += ","
                n.remove(",")

            while "." in n:
                n[n.index(".") - 1] += "."
                n.remove(".")

            t = n

        if flip[0] == "C2":
            if flip[1] == "A":
                t[len(t) - t[::-1].index(prompt["C"]) - 1] = prompt["A"]

        if "IO" in prompt:
            prompt["text"] = " ".join(t)
            flipped_prompts.append(prompt)
        else:
            flipped_prompts.append({"A": prompt["A"], "B": prompt["B"], "C": prompt["C"], "text": " ".join(t)})

    return flipped_prompts


# *Tok Idxs Methods


def get_name_idxs(prompts, tokenizer, idx_types=["IO", "S", "S2"]):
    name_idx_dict = dict((idx_type, []) for idx_type in idx_types)
    for prompt in prompts:
        t = prompt["text"].split(" ")
        toks = tokenizer.tokenize(" ".join(t[:-1]))
        idxs = []
        for idx_type in idx_types:
            if "2" in idx_type:
                idx = len(toks) - toks[::-1].index(tokenizer.tokenize(" " + prompt[idx_type[:-1]])[0]) - 1
            else:
                idx = toks.index(tokenizer.tokenize(" " + prompt[idx_type])[0])

            name_idx_dict[idx_type].append(idx)

    return [torch.tensor(name_idx_dict[idx_type]) for idx_type in idx_types]


def get_end_idxs(prompts, tokenizer, name_tok_len=1):
    toks = torch.Tensor(tokenizer([prompt["text"] for prompt in prompts], padding=True).input_ids).type(torch.int)
    end_idxs = torch.tensor([(toks[i] == 50256).nonzero()[0][0].item() if 50256 in toks[i] else toks.shape[1] for i in range(toks.shape[0])])
    end_idxs = end_idxs - 1 - name_tok_len  # YOURE LOOKING AT TO NOT FINAL IO TOKEN
    return end_idxs


def get_rand_idxs(end_idxs, exclude):
    rand_idxs = []
    for i in range(len(end_idxs)):
        idx = np.random.randint(end_idxs[i])

        while idx in torch.vstack(exclude)[:, i]:
            idx = np.random.randint(end_idxs[i])
        rand_idxs.append(idx)
    return rand_idxs


def get_word_idxs(prompts, word_list, tokenizer):
    """Get the index of the words in word_list in the prompts. Exactly one of the word_list word has to be present in each prompt"""
    idxs = []
    tokenized_words = [tokenizer.tokenize(word)[0] for word in word_list]
    for pr_idx, prompt in enumerate(prompts):
        toks = tokenizer.tokenize(prompt["text"])
        idx = None
        for i, w_tok in enumerate(tokenized_words):
            if word_list[i] in prompt["text"]:
                try:
                    idx = toks.index(w_tok)
                except:
                    raise ValueError(toks, w_tok, prompt["text"])
        if idx is None:
            raise ValueError(f"Word {word_list} and {i} not found {prompt}")
        idxs.append(idx)
    return torch.tensor(idxs)


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ALL_SEM = ["S", "IO","S2", "end", "S+1", "and"]#, "verb", "starts", "S-1", "punct"] # Kevin's antic averages

def get_idx_dict(ioi_prompts, tokenizer):
    (
        IO_idxs,
        S_idxs,
        S2_idxs,
    ) = get_name_idxs(ioi_prompts, tokenizer, idx_types=["IO", "S", "S2"])

    end_idxs = get_end_idxs(ioi_prompts, tokenizer, name_tok_len=1)
    rand_idxs = get_rand_idxs(end_idxs, exclude=[IO_idxs, S_idxs, S2_idxs])
    punc_idxs = get_word_idxs(
        ioi_prompts, [",", "."], tokenizer
    )  # if there is "," and '.' in the prompt, only the '.' index will be kept.
    verb_idxs = get_word_idxs(ioi_prompts, VERBS, tokenizer)
    and_idxs = get_word_idxs(ioi_prompts, [" and"], tokenizer)
    return {
        "IO": IO_idxs,
        "IO-1": IO_idxs-1,
        "IO+1": IO_idxs+1,
        "S": S_idxs,
        "S-1": S_idxs-1,
        "S+1": S_idxs+1,
        "S2": S2_idxs,
        "end": end_idxs,  # the " to" token, the last one.
        "rand": rand_idxs,  # random index at each
        "punct": punc_idxs,
        "verb": verb_idxs,
        "and": and_idxs,
        "starts": torch.zeros_like(and_idxs),
    }


class IOIDataset:
    def __init__(self, prompt_type: str, N=500, tokenizer=None, prompts=None, symmetric=False, prefixes=None):
        assert (prompts is not None) or (not symmetric) or (N%2 == 0), f"{symmetric} {N}"	
        assert prompt_type in ["ABBA", "BABA", "mixed"]
        self.prompt_type = prompt_type
        if prompt_type == "ABBA":
            self.templates = ABBA_TEMPLATES.copy()
        elif prompt_type == "BABA":
            self.templates = BABA_TEMPLATES.copy()
        else:
            self.templates = BABA_TEMPLATES.copy() + ABBA_TEMPLATES.copy()
            random.shuffle(self.templates)
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = tokenizer

        self.prefixes = prefixes
        self.prompt_type = prompt_type
        if prompts is None:
            self.ioi_prompts = gen_prompt_uniform(  # a list of dict of the form {"text": "Alice and Bob bla bla. Bob gave bla to Alice", "IO": "Alice", "S": "Bob"}
                self.templates, NAMES, nouns_dict={"[PLACE]": PLACES, "[OBJECT]": OBJECTS}, N=N, symmetric=symmetric, prefixes=self.prefixes,
            )
        else:
            assert N == len(prompts), f"{N} and {len(prompts)}"
            self.ioi_prompts = prompts

        all_ids = [prompt["T_ID"] for prompt in self.ioi_prompts]
        all_ids_ar = np.array(all_ids)
        self.groups = []
        for id in list(set(all_ids)):
            self.groups.append(np.where(all_ids_ar == id)[0])


        self.text_prompts = [prompt["text"] for prompt in self.ioi_prompts]  # a list of strings

        self.templates_by_prompt = [] #for each prompt if it's ABBA or BABA
        for i in range(N):
            if self.text_prompts[i].index(self.ioi_prompts[i]["IO"]) < self.text_prompts[i].index(self.ioi_prompts[i]["S"]):
                self.templates_by_prompt.append("ABBA")
            else:
                self.templates_by_prompt.append("BABA")

        #print(self.ioi_prompts, "that's that")
        self.toks = torch.Tensor(
            self.tokenizer([prompt["text"] for prompt in self.ioi_prompts], padding=True).input_ids
        ).type(torch.int)

        self.word_idx = get_idx_dict(self.ioi_prompts, self.tokenizer)
        self.sem_tok_idx = {k:v for k, v in self.word_idx.items() if k in ALL_SEM} # the semantic indices that kevin uses
        self.N = N
        self.max_len = max([len(self.tokenizer(prompt["text"]).input_ids) for prompt in self.ioi_prompts])

        self.io_tokenIDs = [self.tokenizer.encode(" " + prompt["IO"])[0] for prompt in self.ioi_prompts]
        self.s_tokenIDs = [self.tokenizer.encode(" " + prompt["S"])[0] for prompt in self.ioi_prompts]

    def gen_flipped_prompts(self, flip):
        """Return a IOIDataset where the name to flip has been replaced by a random name."""
        assert flip in ["S", "S2", "IO"]

        flipped_prompts = gen_flipped_prompts(self.ioi_prompts, NAMES, (flip, "RAND"))
        fliped_ioi_dataset = IOIDataset(
            prompt_type=self.prompt_type, N=self.N, tokenizer=self.tokenizer, prompts=flipped_prompts, prefixes=self.prefixes
        )
        return fliped_ioi_dataset

    def __getitem__(self, key):
        sliced_prompts = self.ioi_prompts[key]
        sliced_dataset=IOIDataset(
            prompt_type=self.prompt_type, N=len(sliced_prompts), tokenizer=self.tokenizer, prompts=sliced_prompts, prefixes=self.prefixes
        )
        return sliced_dataset
        
    def __setitem__(self, key, value):
        raise NotImplementedError()

    def __delitem__(self, key):
        raise NotImplementedError()

def clear_gpu_mem():
    gc.collect()
    torch.cuda.empty_cache()


def show_tokens(tokens, return_list=False):
    # Prints the tokens as text, separated by |
    if type(tokens) == str:
        # If we input text, tokenize first
        tokens = model.to_tokens(tokens)
    text_tokens = [model.tokenizer.decode(t) for t in tokens.squeeze()]
    if return_list:
        return text_tokens
    else:
        print("|".join(text_tokens))


def show_pp(m, xlabel="", ylabel="", title="", bartitle=""):
    """
    Plot a heatmap of the values in the matrix `m`
    """
    fig = px.imshow(
        m.T,
        title=title if title else "",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
    )

    fig.update_layout(
        coloraxis_colorbar=dict(
            title=bartitle,
            thicknessmode="pixels",
            thickness=50,
            lenmode="pixels",
            len=300,
            yanchor="top",
            y=1,
            ticks="outside",
        ),
        xaxis_title="",
    )

    fig.update_layout(yaxis_title=ylabel, xaxis_title=xlabel)
    fig.show()


# Plot attention patterns weighted by value norm


def show_attention_patterns(model, heads, texts, mode="val", title_suffix=""):
    assert mode in ["attn", "val"]  # value weighted attention or attn for attention probas
    assert type(texts) == list

    for (layer, head) in heads:
        cache = {}


        good_names = [f"blocks.{layer}.attn.hook_attn"]
        if mode == "val":
            good_names.append(f"blocks.{layer}.attn.hook_v")
        model.cache_some(cache=cache, names=lambda x: x in good_names)  # shape: batch head_no seq_len seq_len

        logits = model(texts)

        for i, text in enumerate(texts):
            assert len(list(cache.items())) == 1 + int(mode == "val"), len(list(cache.items()))
            toks = model.tokenizer(text)["input_ids"]
            words = [model.tokenizer.decode([tok]) for tok in toks]
            attn = cache[good_names[0]].detach().cpu()[i, head, :, :]
            if mode == "val":
                vals = cache[good_names[1]].detach().cpu()[i, :, head, :].norm(dim=-1)
                cont = torch.einsum("ab,b->ab", attn, vals)

            fig = px.imshow(
                attn if mode == "attn" else cont,
                title=f"{layer}.{head} Attention" + title_suffix,
                color_continuous_midpoint=0,
                color_continuous_scale="RdBu",
                labels={"y": "Queries", "x": "Keys"},
            )

            fig.update_layout(
                xaxis={"side": "top", "ticktext": words, "tickvals": list(range(len(words))), "tickfont": dict(size=8)},
                yaxis={"ticktext": words, "tickvals": list(range(len(words))), "tickfont": dict(size=8)},
            )

            fig.show()

def safe_del(a):
    """Try and delete a even if it doesn't yet exist"""
    try:
        exec(f"del {a}")
    except:
        pass
    torch.cuda.empty_cache()
#%% [markdown]
# # Copying experiments.
# CLAIM: heads 9.6, 9.9 and 10.0 write the IO into the residual stream, by attending to that token and copying it.
# For now we review why we think this is true.
# Experiments for minimality and completeness will follow in the next section (faithfulness). # TODO is the claim too strong? 
#%% # plot writing in the IO - S direction
model_name = "gpt2"
safe_del("model")
print_gpu_mem("About to load model")
model = EasyTransformer(model_name, use_attn_result=True) #use_attn_result adds a hook blocks.{lay}.attn.hook_result that is before adding the biais of the attention layer
device = "cuda"
if torch.cuda.is_available():
    model.to(device)
print_gpu_mem("Gpt2 loaded")

ioi_dataset = IOIDataset(prompt_type = "mixed", N=200, tokenizer=model.tokenizer) 
ioi_prompts = ioi_dataset.ioi_prompts

webtext = load_dataset("stas/openwebtext-10k")
owb_seqs = ["".join(show_tokens(webtext['train']['text'][i][:2000], return_list=True)[:ioi_dataset.max_len]) for i in range(ioi_dataset.N)]

#%%
def writing_direction_heatmap(
    model,
    prompts, 
    mode="attn_out", 
    return_vals=False, 
    dir_mode = "IO - S",
    unembed_mode = "normal", # or "Neel"
    title="",
):
    """
    Plot the dot product between how much each attention head
    output with `IO-S`, the difference between the unembeds between
    the (correct) IO token and the incorrect S token
    """

    n_heads = model.cfg["n_heads"]
    n_layers = model.cfg["n_layers"]

    model_unembed = model.unembed.W_U.detach().cpu() #note that for GPT2 embeddings and unembeddings are tides such that W_E = Transpose(W_U)

    if mode == "attn_out": # heads, layers
        vals = torch.zeros(size=(n_heads, n_layers))
    elif mode == "mlp":
        vals = torch.zeros(size=(1, n_layers))
    else:
        raise NotImplementedError()

    N = len(prompts)
    for prompt in tqdm(prompts):
        io_tok = model.tokenizer(" "+prompt["IO"])["input_ids"][0]
        s_tok = model.tokenizer(" "+prompt["S"])["input_ids"][0]
        io_dir = model_unembed[io_tok]
        s_dir = model_unembed[s_tok]
        if dir_mode == "IO - S":
            dire = io_dir - s_dir
        elif dir_mode == "IO":
            dire = io_dir
        elif dir_mode == "S":
            dire = s_dir
        else:
            raise NotImplementedError()

        model.reset_hooks()
        cache = {}
        model.cache_all(cache)

        logits = model(prompt["text"])

        for lay in range(n_layers):
            if mode == "attn_out": 
                cur = cache[f"blocks.{lay}.attn.hook_result"][0,-2,:,:]
            elif mode == "mlp":
                cur = cache[f"blocks.{lay}.hook_mlp_out"][:,-2,:]
            vals[:,lay] += torch.einsum("ha,a->h", cur.cpu(), dire.cpu())

    vals /= N
    show_pp(vals, xlabel="head no", ylabel="layer no", title=title)
    if return_vals: return vals

attn_vals = writing_direction_heatmap(
    model,
    ioi_dataset.ioi_prompts[:51], 
    return_vals=True, 
    mode="attn_out", 
    dir_mode="IO - S", 
    title="Attention head output into IO - S token unembedding (GPT2)",
)
#%% # check that this attending to IO happens as described
show_attention_patterns(model, [(9,9), (9,6), (10,0)], ioi_dataset.text_prompts[:1])
#%% # the more attention, the more writing
def scatter_attention_and_contribution(
    model,
    layer_no,
    head_no,
    prompts,
    gpt_model="gpt2",
    return_vals=False,
):
    """
    Plot a scatter plot 
    for each input sequence with the attention paid to IO and S
    and the amount that is written in the IO and S directions
    """
    n_heads = model.cfg["n_heads"]
    n_layers = model.cfg["n_layers"]
    model_unembed = model.unembed.W_U.detach().cpu()
    N = len(prompts)
    df = []
    for prompt in tqdm(prompts):
        io_tok = model.tokenizer(" "+prompt["IO"])["input_ids"][0]
        s_tok = model.tokenizer(" "+prompt["S"])["input_ids"][0]
        toks = model.tokenizer(prompt["text"])["input_ids"]
        io_pos = toks.index(io_tok)
        s1_pos = toks.index(s_tok)
        s2_pos = toks[s1_pos+1:].index(s_tok) + (s1_pos+1)
        assert toks[-1] == io_tok

        io_dir = model_unembed[io_tok].detach().cpu()
        s_dir = model_unembed[s_tok].detach().cpu()

        model.reset_hooks()
        cache = {}
        model.cache_all(cache)

        logits = model(prompt["text"])

        for dire, posses, tok_type in [(io_dir, [io_pos], "IO"), (s_dir, [s1_pos, s2_pos], "S")]:
            prob = sum([cache[f"blocks.{layer_no}.attn.hook_attn"][0, head_no, -2, pos].detach().cpu() for pos in posses])
            resid = cache[f"blocks.{layer_no}.attn.hook_result"][0, -2, head_no, :].detach().cpu()
            dot = torch.einsum("a,a->", resid, dire)
            df.append([prob, dot, tok_type, prompt["text"]])

    # most of the pandas stuff is intuitive, no need to deeply understand
    viz_df = pd.DataFrame(df, columns = [f"Attn Prob on Name", f"Dot w Name Embed", "Name Type", "text"])
    fig = px.scatter(viz_df, x = f"Attn Prob on Name", y = f"Dot w Name Embed", color = "Name Type", hover_data = ["text"], title = f"How Strong {layer_no}.{head_no} Writes in the Name Embed Direction Relative to Attn Prob")
    fig.show()
    if return_vals: return viz_df

scatter_attention_and_contribution(model, 9, 9, ioi_prompts[:500], gpt_model="gpt2")
scatter_attention_and_contribution(model, 9, 6, ioi_prompts[:500], gpt_model="gpt2")
scatter_attention_and_contribution(model, 10, 0, ioi_prompts[:500], gpt_model="gpt2")
#%% # for control purposes, check that there is unlikely to be a correlation between attention and writing for unimportant heads
scatter_attention_and_contribution(model, random.randint(0,11), random.randint(0,11), ioi_prompts[:500], gpt_model="gpt2") 
#%%
# TODO diff circuits with templates experiment
#%% [markdown]
# To ensure that the name movers heads are indeed only copying information, we conduct a "check copying circuit" experiment. This means that we only keep the first layer of the transformer and apply the OV circuit of the head and decode the logits from that. Every other component of the transformer is deleted (i.e. zero ablated). 
#%%
def check_copy_circuit(model, layer, head, ioi_dataset, verbose=False):
    cache = {}
    model.cache_some(cache, lambda x: x=="blocks.0.hook_resid_post")
    model(ioi_dataset.text_prompts)
    z_0 = model.blocks[1].ln1( cache["blocks.0.hook_resid_post"])
    v = (z_0@model.blocks[layer].attn.W_V[head].T + model.blocks[layer].attn.b_V[head])
    o = torch.einsum('sph,dh->spd', v, model.blocks[layer].attn.W_O[head])
    logits = model.unembed(model.ln_final(o))
    k=5
    n_right = 0
    

    for seq_idx,prompt in enumerate(ioi_dataset.ioi_prompts):
        #print(prompt)
        for word in ["IO", "S", "S2"]:
            pred_tokens = [model.tokenizer.decode(token) for token in torch.topk(logits[seq_idx, ioi_dataset.word_idx[word][seq_idx]], k).indices]
            if "S" in word:
                name = "S"
            else:
                name = word
            if " "+prompt[name] in pred_tokens:
                n_right+=1
            else:
                if verbose:
                    print("-------")
                    print("Seq: " + ioi_dataset.text_prompts[seq_idx])
                    print("Target: " + ioi_dataset.ioi_prompts[seq_idx][name])
                    print(' '.join([f'({i+1}):{model.tokenizer.decode(token)}' for i, token in enumerate(torch.topk(logits[seq_idx, ioi_dataset.word_idx[word][seq_idx]], k).indices)]))
    percent_right = (n_right / (ioi_dataset.N*3)) * 100
    print(f'Copy circuit for head {layer}.{head} : Top {k} accuracy: {percent_right}%')

print(" --- Name Mover heads --- ")
check_copy_circuit(model, 9,9,ioi_dataset)
check_copy_circuit(model, 10,0,ioi_dataset)
check_copy_circuit(model, 9,6,ioi_dataset)

print(" --- Calibration heads --- ")
check_copy_circuit(model, 10,7,ioi_dataset)
check_copy_circuit(model, 11,10,ioi_dataset)

print(" ---  Random heads for control ---  ")
check_copy_circuit(model, random.randint(0,11), random.randint(0,11),ioi_dataset) 
check_copy_circuit(model, random.randint(0,11), random.randint(0,11),ioi_dataset) 
check_copy_circuit(model, random.randint(0,11), random.randint(0,11),ioi_dataset) 
#%% [markdown]
# For calibration heads, we observe a reverse trend to name movers, the more is pays attention to a name, the more it write in its *oposite* direction. Why is that? 
# You need to remember the training objective of the transformer: it has to predict accurate probability distribution over all the next tokens. 
# If previously it was able to recover the IO, in the final layer it has to callibrate the probability of this particular token, it cannot go all in "THE NEXT TOKEN IS BOB" with 100% proba.
# This is why we observe calibration mechanisms that do back and forth and seems to inhibate information put by earlier modules.

# You can see this similarly as open loop / closed loop optimization. It's easier to make a good guess by making previous rough estimate more precise than making a good guess in one shot.
#%%
scatter_attention_and_contribution(model, 10, 7, ioi_prompts[:500], gpt_model="gpt2")
scatter_attention_and_contribution(model, 11, 10, ioi_prompts[:500], gpt_model="gpt2")
#%% [markdown]
# # Faithfulness: ablating everything but the circuit
# For each template, e.g `Then, [A] and [B] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to [A]` we ablate only the indices we claim are important and we retain a positive logit difference between `IO` and `S`, as well the "score" (whether the IO logit remains in the top 10 logit AND IO > S), though have some performace degradation, particularly when we don't ablate the name movers heads.
#%% # run normal ablation experiments
num_templates = 10 # len(ABBA_TEMPLATES)
template_type = "BABA"
if template_type == "ABBA":
    templates = ABBA_TEMPLATES[:num_templates]
elif template_type == "BABA":
    templates = BABA_TEMPLATES[:num_templates]
else:
    raise NotImplementedError()

def logit_diff(model, ioi_dataset, all=False, std=False):
    """Difference between the IO and the S logits at the "to" token"""
    logits = model(ioi_dataset.text_prompts).detach()
    L = len(ioi_dataset.text_prompts)
    IO_logits = logits[
        torch.arange(len(ioi_dataset.text_prompts)), ioi_dataset.word_idx["end"][:L], ioi_dataset.io_tokenIDs[:L]
    ]
    S_logits = logits[
        torch.arange(len(ioi_dataset.text_prompts)), ioi_dataset.word_idx["end"][:L], ioi_dataset.s_tokenIDs[:L]
    ]

    print("LOGIT_DIFF:", IO_logits - S_logits)

    if all and not std:
        return IO_logits - S_logits
    if std:
        if all:
            first_bit = IO_logits - S_logits
        else:
            first_bit = (IO_logits - S_logits).mean().detach().cpu()
        return first_bit, torch.std(IO_logits - S_logits)
    return (IO_logits - S_logits).mean().detach().cpu()

def score(model, ioi_dataset, all=False):
    text_prompts = ioi_dataset.text_prompts
    logits = model(text_prompts).detach()
    L = len(text_prompts)
    end_logits = logits[
        torch.arange(len(text_prompts)), ioi_dataset.word_idx["end"][:L], :
    ]  # batch * sequence length * vocab_size
    io_logits = end_logits[torch.arange(len(text_prompts)), ioi_dataset.io_tokenIDs[:L]]
    assert len(list(end_logits.shape)) == 2, end_logits.shape
    top_10s_standard = torch.topk(end_logits, dim=1, k=10).values[:, -1]
    good_enough = end_logits > top_10s_standard.unsqueeze(-1)
    selected_logits = good_enough[torch.arange(len(text_prompts)), ioi_dataset.io_tokenIDs[:L]]

    # is IO > S ???
    IO_logits = logits[torch.arange(len(text_prompts)), ioi_dataset.word_idx["end"][:L], ioi_dataset.io_tokenIDs[:L]]
    S_logits = logits[torch.arange(len(text_prompts)), ioi_dataset.word_idx["end"][:L], ioi_dataset.s_tokenIDs[:L]]
    IO_greater_than_S = (IO_logits - S_logits) > 0

    # calculate percentage passing both tests
    answer = torch.sum((selected_logits & IO_greater_than_S).float()).detach().cpu() / len(text_prompts)

    selected = torch.sum(selected_logits) / len(text_prompts)
    greater = torch.sum(IO_greater_than_S) / len(text_prompts)

    print(f"Score calc: {answer}; {selected} and {greater}")
    return answer

def io_probs(model, ioi_dataset, mode="IO"):  # also S mode
    assert mode in ["IO", "S"]
    text_prompts = ioi_dataset.text_prompts
    logits = model(text_prompts).detach()
    assert len(list(logits.shape)) == 3, logits.shape
    L = len(text_prompts)
    assert logits.shape[0] == L
    end_logits = logits[
        torch.arange(len(text_prompts)), ioi_dataset.word_idx["end"][:L], :
    ]  # batch * sequence length * vocab_size
    end_probs = torch.softmax(end_logits, dim=-1)
    ids = ioi_dataset.io_tokenIDs if mode == "IO" else ioi_dataset.s_tokenIDs
    probs = end_probs[torch.arange(len(text_prompts)), ids[:L]]
    return probs.mean().detach().cpu()


N = 10  # number per template
template_prompts = [
    gen_prompt_uniform(
        templates[i : i + 1],
        NAMES,
        NOUNS_DICT,
        N=N,
        symmetric=False,
    )
    for i in range(len(templates))
]

for ablate_calibration in [False, True]:
    ld_data = []
    score_data = []
    probs_data = []
    sprobs_data = []
    io_logits_data = []
    s_logits_data = []
    for template_idx in tqdm(range(num_templates)):
        prompts = template_prompts[template_idx]
        ioi_dataset = IOIDataset(prompt_type=template_type, N=N, symmetric=False, prompts=prompts)
        assert torch.all(ioi_dataset.toks != 50256)  # no padding anywhere
        assert len(ioi_dataset.sem_tok_idx.keys()) != 0, "no semantic tokens found"
        for key in ioi_dataset.sem_tok_idx.keys():
            idx = ioi_dataset.sem_tok_idx[key][0]
            assert torch.all(ioi_dataset.sem_tok_idx[key] == idx), f"{key} {ioi_dataset.sem_tok_idx[key]}"
            # check that semantic ablation = normal ablation

        try:
            del model
            torch.cuda.empty_cache()
        except:
            pass
        model = EasyTransformer("gpt2", use_attn_result=True).to(device)

        seq_len = ioi_dataset.toks.shape[1]
        head_indices_to_ablate = {
            (i % 12, i // 12): [list(range(seq_len)) for _ in range(len(ioi_dataset.text_prompts))] for i in range(12 * 12)
        }

        mlp_indices_to_ablate = [[] for _ in range(model.cfg["n_heads"])]

        for head in [
            (0, 1),
            (0, 10),
            (3, 0),
        ]:
            head_indices_to_ablate[head] = [i for i in range(seq_len) if i != ioi_dataset.sem_tok_idx["S2"][0]]

        for head in [
            (4, 11),
            (2, 2),
            (2, 9),
        ]:
            head_indices_to_ablate[head] = [
                i for i in range(seq_len) if i not in [ioi_dataset.sem_tok_idx["S"][0], ioi_dataset.sem_tok_idx["and"][0]]
            ]

        for head in [
            (5, 8),
            (5, 9),
            (5, 5),
            (6, 9),
        ]:
            head_indices_to_ablate[head] = [i for i in range(seq_len) if i not in [ioi_dataset.sem_tok_idx["S2"][0]]]

        end_heads = [
            (7, 3),
            (7, 9),
            (8, 6),
            (8, 10),
            (9, 6),
            (9, 9),
            (10, 0),        
        ]

        if ablate_calibration:
            end_heads += [(10, 7), (11, 12)]

        for head in end_heads:
            head_indices_to_ablate[head] = [i for i in range(seq_len) if i not in [ioi_dataset.sem_tok_idx["end"][0]]]

        # define the ablation function for ALL parts of the model at once
        def ablation(z, mean, hook):
            layer = int(hook.name.split(".")[1])
            head_idx = hook.ctx["idx"]
            head = (layer, head_idx)

            if "mlp_out" in hook.name:
                # ablate the relevant parts
                for i in range(z.shape[0]):
                    z[i, mlp_indices_to_ablate[layer]] = mean[i, mlp_indices_to_ablate[layer]].to(z.device)

            if "attn.hook_result" in hook.name:  # and (layer, head) not in heads_to_keep:
                # ablate
                assert len(z.shape) == 3, z.shape  # we specifically get sent the relevant head
                assert 12 not in list(z.shape), "Yikes, probably dim kept back is wrong, should be head dim"

                # see above
                for i in range(z.shape[0]):
                    z[i, head_indices_to_ablate[head]] = mean[i, head_indices_to_ablate[head]].to(z.device)

            return z

        ld_metric = ExperimentMetric(metric=logit_diff, dataset=ioi_dataset, relative_metric=False)
        score_metric = ExperimentMetric(metric=score, dataset=ioi_dataset, relative_metric=False)
        ld_metric.set_baseline(model)
        score_metric.set_baseline(model)
        probs_metric = ExperimentMetric(metric=io_probs, dataset=ioi_dataset, relative_metric=False)
        probs_metric.set_baseline(model)
        sprobs_metric = ExperimentMetric(
            metric=lambda x, y: io_probs(x, y, mode="S"), dataset=ioi_dataset, relative_metric=False
        )
        sprobs_metric.set_baseline(model)
        io_logits_metric = ExperimentMetric(
            metric=lambda x, y: logit_diff(x, y, all=True)[0], dataset=ioi_dataset, relative_metric=False
        )
        io_logits_metric.set_baseline(model)
        s_logits_metric = ExperimentMetric(
            metric=lambda x, y: logit_diff(x, y, all=True)[1], dataset=ioi_dataset, relative_metric=False
        )
        s_logits_metric.set_baseline(model)

        config = AblationConfig(
            abl_type="custom",
            abl_fn=ablation,
            mean_dataset=ioi_dataset.text_prompts,
            target_module="attn_head",
            head_circuit="result",
            cache_means=True,
            verbose=True,
        )

        abl = EasyAblation(model, config, ld_metric)  # , semantic_indices=ioi_dataset.sem_tok_idx) # semantic indices should not be necessary

        model.reset_hooks()
        for layer in range(12):
            for head in range(12):
                model.add_hook(*abl.get_hook(layer, head))
            model.add_hook(*abl.get_hook(layer, head=None, target_module="mlp"))

        ld = ld_metric.compute_metric(model)
        print(f"{ld=}")
        ld_data.append((ld_metric.baseline, ld))

        cur_score = score_metric.compute_metric(model)
        print(f"{cur_score=}")
        score_data.append((score_metric.baseline, cur_score))

        cur_probs = probs_metric.compute_metric(model)
        print(f"{cur_probs=}")
        probs_data.append((probs_metric.baseline, cur_probs))

        s_probs = sprobs_metric.compute_metric(model)  # s probs is like 0.003 for most ablate calibration heads # or is is that low
        print(f"{s_probs=}")
        sprobs_data.append((sprobs_metric.baseline, s_probs))

        io_logits = io_logits_metric.compute_metric(model)
        print(f"{io_logits=}")
        io_logits_data.append((io_logits_metric.baseline, io_logits))

        s_logits = s_logits_metric.compute_metric(model)
        print(f"{s_logits=}")
        s_logits_data.append((s_logits_metric.baseline, s_logits))

    plotly.io.renderers.default = "notebook"

    xs = [ld_data[i][0].item() for i in range(num_templates)]
    ys = [ld_data[i][1].item() for i in range(num_templates)]

    x_label = "Baseline Logit Diff" # IO Probability"
    y_label = "Ablated Logit Diff" # IO Probability"

    d = {
        x_label: xs,
        y_label: ys,
    }
    d["beg"] = [template[:10] for template in templates]
    d["sentence"] = [template for template in templates]
    d["commas"] = [template.count(",") for template in templates]

    df = pd.DataFrame(d)
    px.scatter(df, x=x_label, y=y_label, hover_data=["sentence"], text="beg", title=f"Change in logit diff when {ablate_calibration=}").show()
#%% [markdown]
# # Circuit extraction experiments 
# #%% # Dataset initialisation
# N=100
# ioi_dataset = IOIDataset(prompt_type="mixed", N=N, symmetric=True, prefixes=None) #["Two friends were discussing.", "It was a levely day.", "Two friends arrived in a new place.", "The couple arrived."]) # , prompts=saved_prompts) # [{"IO" : "Anthony", "S" : "Aaron", "text" : "Then, Aaron and Anthony went to the grocery store. Aaron gave a ring to Anthony"}, {'IO': 'Lindsey', 'S': 'Joshua', 'text': 'Then, Joshua and Lindsey were working at the grocery store. Joshua decided to give a basketball to Lindsey'}], symmetric=True)
# ioi_prompts = ioi_dataset.ioi_prompts
# pprint(ioi_dataset.text_prompts[:5])  # example prompts
#%%
ld = logit_diff(model, ioi_dataset[:N], all=True)

def list_diff(l1, l2):
    l2_ = [int(x) for x in l2]
    return list(set(l1).difference(set(l2_)))


def turn_keep_in_rmv(to_keep, max_len):
    to_rmv = {}
    for t in to_keep.keys():
        to_rmv[t] = []
        for idxs in to_keep[t]:
            to_rmv[t].append(list_diff(list(range(max_len)), idxs))
    return to_rmv


def process_heads_and_mlps(
    heads_to_remove=None,  # {(2,3) : List[List[int]]: dimensions dataset_size * datapoint_length
    mlps_to_remove=None,  # {2: List[List[int]]: dimensions dataset_size * datapoint_length
    heads_to_keep=None,  # as above for heads
    mlps_to_keep=None,  # as above for mlps
    no_prompts=0,
    ioi_dataset=None,
    model=None,
):
    assert (heads_to_remove is None) != (heads_to_keep is None)
    assert (mlps_to_keep is None) != (mlps_to_remove is None)
    assert no_prompts == len(ioi_dataset.text_prompts)

    n_layers = model.cfg["n_layers"]
    n_heads = model.cfg["n_heads"]

    if mlps_to_remove is not None:
        mlps = mlps_to_remove.copy()
    else:  # do smart computation in mean cache
        mlps = mlps_to_keep.copy()
        for l in range(n_layers):
            if l not in mlps_to_keep:
                mlps[l] = [[] for _ in range(no_prompts)]
        mlps = turn_keep_in_rmv(
            mlps, ioi_dataset.max_len
        )  # TODO check that this is still right for the max_len of maybe shortened datasets

    if heads_to_remove is not None:
        heads = heads_to_remove.copy()
    else:
        heads = heads_to_keep.copy()
        for l in range(n_layers):
            for h in range(n_heads):
                if (l, h) not in heads_to_keep:
                    heads[(l, h)] = [[] for _ in range(no_prompts)]
        heads = turn_keep_in_rmv(heads, ioi_dataset.max_len)
    return heads, mlps
    # print(mlps, heads)


def get_circuit_replacement_hook(
    heads_to_remove=None,
    mlps_to_remove=None,
    heads_to_keep=None,
    mlps_to_keep=None,
    heads_to_remove2=None,
    mlps_to_remove2=None,
    heads_to_keep2=None,
    mlps_to_keep2=None,
    no_prompts=0,
    ioi_dataset=None,
    model=None,
):
    heads, mlps = process_heads_and_mlps(
        heads_to_remove=heads_to_remove,  # {(2,3) : List[List[int]]: dimensions dataset_size * datapoint_length
        mlps_to_remove=mlps_to_remove,  # {2: List[List[int]]: dimensions dataset_size * datapoint_length
        heads_to_keep=heads_to_keep,  # as above for heads
        mlps_to_keep=mlps_to_keep,  # as above for mlps
        no_prompts=no_prompts,
        ioi_dataset=ioi_dataset,
        model=model,
    )

    if (heads_to_remove2 is not None) or (heads_to_keep2 is not None):
        heads2, mlps2 = process_heads_and_mlps(
            heads_to_remove=heads_to_remove2,  # {(2,3) : List[List[int]]: dimensions dataset_size * datapoint_length
            mlps_to_remove=mlps_to_remove2,  # {2: List[List[int]]: dimensions dataset_size * datapoint_length
            heads_to_keep=heads_to_keep2,  # as above for heads
            mlps_to_keep=mlps_to_keep2,  # as above for mlps
            no_prompts=no_prompts,
            ioi_dataset=ioi_dataset,
            model=model,
        )
    else:
        heads2, mlps2 = heads, mlps

    def circuit_replmt_hook(z, act, hook):  # batch, seq, heads, head dim
        layer = int(hook.name.split(".")[1])
        if "mlp" in hook.name and layer in mlps:
            for i in range(no_prompts):
                z[i, mlps[layer][i], :] = act[
                    i, mlps2[layer][i], :
                ]  # ablate all the indices in mlps[layer][i]; mean may contain semantic ablation
                # TODO can this i loop be vectorized?

        if "attn.hook_result" in hook.name and (layer, hook.ctx["idx"]) in heads:
            for i in range(no_prompts):  # we use the idx from contex to get the head
                z[i, heads[(layer, hook.ctx["idx"])][i], :] = act[i, heads2[(layer, hook.ctx["idx"])][i], :]

        return z

    return circuit_replmt_hook, heads, mlps


def do_circuit_extraction(
    heads_to_remove=None,  # {(2,3) : List[List[int]]: dimensions dataset_size * datapoint_length
    mlps_to_remove=None,  # {2: List[List[int]]: dimensions dataset_size * datapoint_length
    heads_to_keep=None,  # as above for heads
    mlps_to_keep=None,  # as above for mlps
    no_prompts=0,
    ioi_dataset=None,
    model=None,
):
    """
    if `ablate` then ablate all `heads` and `mlps`
        and keep everything else same
    otherwise, ablate everything else
        and keep `heads` and `mlps` the same
    """
    # check if we are either in keep XOR remove move from the args
    ablation, heads, mlps = get_circuit_replacement_hook(
        heads_to_remove=heads_to_remove,  # {(2,3) : List[List[int]]: dimensions dataset_size * datapoint_length
        mlps_to_remove=mlps_to_remove,  # {2: List[List[int]]: dimensions dataset_size * datapoint_length
        heads_to_keep=heads_to_keep,  # as above for heads
        mlps_to_keep=mlps_to_keep,  # as above for mlps
        no_prompts=no_prompts,
        ioi_dataset=ioi_dataset,
        model=model,
    )

    metric = ExperimentMetric(
        metric=logit_diff, dataset=ioi_dataset.text_prompts[:no_prompts], relative_metric=False
    )  # TODO make dummy metric

    config = AblationConfig(
        abl_type="custom",
        abl_fn=ablation,
        mean_dataset=ioi_dataset.text_prompts[:no_prompts],  # TODO nb of prompts useless ?
        target_module="attn_head",
        head_circuit="result",
        cache_means=True,  # circuit extraction *has* to cache means. the get_mean reset the
        verbose=True,
    )
    abl = EasyAblation(
        model,
        config,
        metric,
        semantic_indices=ioi_dataset.sem_tok_idx,
        mean_by_groups=True,  # TO CHECK CIRCUIT BY GROUPS
        groups=ioi_dataset.groups,
        blue_pen=False,
    )
    model.reset_hooks()

    for layer, head in heads.keys():
        model.add_hook(*abl.get_hook(layer, head))
    for layer in mlps.keys():
        model.add_hook(*abl.get_hook(layer, head=None, target_module="mlp"))

    return model, abl
# %% # sanity check

if False:
    type(ioi_dataset)
    old_ld = logit_diff(model, ioi_dataset[:N])
    model, abl_cricuit_extr = do_circuit_extraction(
        heads_to_remove={
            (0, 4): [list(range(ioi_dataset.max_len)) for _ in range(N)]
        },  # annoyingly sometimes needs to be edited...
        mlps_to_remove={},
        heads_to_keep=None,
        mlps_to_keep=None,
        no_prompts=N,
        model=model,
        ioi_dataset=ioi_dataset[:N],
    )
    ld = logit_diff(model, ioi_dataset[:N])
    metric = ExperimentMetric(metric=logit_diff, dataset=ioi_dataset.text_prompts[:N], relative_metric=False)
    config = AblationConfig(
        abl_type="mean",
        mean_dataset=ioi_dataset.text_prompts[:N],
        target_module="attn_head",
        head_circuit="result",
        cache_means=True,
    )  #  abl_fn=mean_at_end) # mean_dataset=owb_seqs, target_module="mlp", head_circuit="result", cache_means=True, verbose=True)
    abl = EasyAblation(
        model,
        config,
        metric,
        semantic_indices=ioi_dataset[:N].sem_tok_idx,
        mean_by_groups=True,  # TO CHECK CIRCUIT BY GROUPS
        groups=ioi_dataset.groups,
        blue_pen=False,
    )
    res = abl.run_experiment()
    print(ld, res[:5, :5])


#%%
def score_metric(model, ioi_dataset, K=1, target_dataset=None,all=False):
    if target_dataset is None:
        target_dataset = ioi_dataset
    text_prompts = ioi_dataset.text_prompts
    logits = model(text_prompts).detach()
    #print(get_corner(logits))
    #print(text_prompts[:2])
    L = len(text_prompts)
    end_logits = logits[
        torch.arange(len(text_prompts)), ioi_dataset.word_idx["end"][:L], :
    ]  # batch * sequence length * vocab_size
    io_logits = end_logits[torch.arange(len(text_prompts)), target_dataset.io_tokenIDs[:L]]
    assert len(list(end_logits.shape)) == 2, end_logits.shape
    top_10s_standard = torch.topk(end_logits, dim=1, k=K).values[:, -1]
    good_enough = end_logits >= top_10s_standard.unsqueeze(-1)
    selected_logits = good_enough[torch.arange(len(text_prompts)), target_dataset.io_tokenIDs[:L]]
    #print(torch.argmax(end_logits, dim=-1))
    # is IO > S ???
    IO_logits = logits[torch.arange(len(text_prompts)), ioi_dataset.word_idx["end"][:L], target_dataset.io_tokenIDs[:L]]
    S_logits = logits[torch.arange(len(text_prompts)), ioi_dataset.word_idx["end"][:L], target_dataset.s_tokenIDs[:L]]
    IO_greater_than_S = (IO_logits - S_logits) > 0

    # calculate percentage passing both tests
    answer = torch.sum((selected_logits & IO_greater_than_S).float()).detach().cpu() / len(text_prompts)

    selected = torch.sum(selected_logits) / len(text_prompts)
    greater = torch.sum(IO_greater_than_S) / len(text_prompts)

    #print(f"Kevin gives: {answer}; {selected} and {greater}")
    return answer


def print_top_k(model, ioi_dataset, K=1, n=10):
    logits = model(ioi_dataset.text_prompts).detach()
    end_logits = logits[
        torch.arange(len(ioi_dataset.text_prompts)), ioi_dataset.word_idx["end"], :
    ]  # batch * sequence length * vocab_size
    probs = np.around(torch.nn.functional.log_softmax(end_logits, dim=-1).cpu().numpy(), 2)
    topk = torch.topk(end_logits, dim=1, k=K).indices
    for x in range(n):
        print("-------------------")
        print(ioi_dataset.text_prompts[x])
        print(' '.join([f'({i+1}):{model.tokenizer.decode(token)} : {probs[x][token]}' for i, token in enumerate(topk[x])]))
# %%
print_top_k(model, ioi_dataset, K=5)

# %%  Running circuit extraction

def join_lists(l1, l2): # l1 is a list of list. l2 a list of int. We add the int from l2 to the lists of l1.
    assert len(l1) == len(l2)
    assert type(l1[0]) == list  and type(l2[0]) == int
    l = []
    for i in range(len(l1)):
        l.append(l1[i]+[l2[i]])
    return l

def get_extracted_idx(idx_list: list[str], no_prompts, ioi_dataset):
    int_idx = [[] for i in range(no_prompts)]
    for idx_name in idx_list:
        int_idx_to_add = [int(x) for x in list(ioi_dataset.word_idx[idx_name][:no_prompts])] #torch to python objects
        int_idx = join_lists(int_idx, int_idx_to_add)
    return int_idx

def get_heads_circuit(ioi_dataset, no_prompts, calib_head=True, mlp0=False):
    heads_to_keep = {}
    heads_to_keep[(0, 1)] = get_extracted_idx(["S2"], no_prompts, ioi_dataset)
    heads_to_keep[(0, 10)] = get_extracted_idx(
        ["S2"], no_prompts, ioi_dataset
    )  # torch.hstack([S_idxs.unsqueeze(1), S2_idxs.unsqueeze(1)])
    heads_to_keep[(3, 0)] = get_extracted_idx(["S2"], no_prompts, ioi_dataset)
    heads_to_keep[(4, 11)] = get_extracted_idx(["S+1", "and"], no_prompts, ioi_dataset)
    heads_to_keep[(2, 2)] = get_extracted_idx(["S+1", "and"], no_prompts, ioi_dataset)
    heads_to_keep[(2, 9)] = get_extracted_idx(["S+1", "and"], no_prompts, ioi_dataset)
    heads_to_keep[(5, 8)] = get_extracted_idx(["S2"], no_prompts, ioi_dataset)
    heads_to_keep[(5, 9)] = get_extracted_idx(["S2"], no_prompts, ioi_dataset)
    heads_to_keep[(5, 5)] = get_extracted_idx(["S2"], no_prompts, ioi_dataset)
    heads_to_keep[(6, 9)] = get_extracted_idx(["S2"], no_prompts, ioi_dataset)
    for (h, l) in [
        (7, 3),
        (7, 9),
        (8, 6),
        (8, 10),
        (9, 6),
        (9, 9),
        (10, 0),
    ]:  # , (10,7), (11, 10)]:#, (10,7), (11, 10)]:
        heads_to_keep[(h, l)] = get_extracted_idx(["end"], no_prompts, ioi_dataset)
    if calib_head:
        for (h, l) in [(10, 7), (11, 10)]:
            heads_to_keep[(h, l)] = get_extracted_idx(["end"], no_prompts, ioi_dataset)
    if mlp0:
        mlps_to_keep = {}
        mlps_to_keep[0] = get_extracted_idx(
            ["IO", "and", "S", "S+1", "S2", "end"], no_prompts, ioi_dataset
        )  # IO, AND, S, S+1, S2, and END
        return heads_to_keep, mlps_to_keep
    return heads_to_keep

no_prompts = N
heads_to_keep = get_heads_circuit(ioi_dataset, no_prompts)

mlps_to_keep={}
#mlps_to_keep[0] = [list(range(ioi_dataset.max_len)) for i in range(no_prompts)]

model.reset_hooks()
old_ld, old_std = logit_diff(model, ioi_dataset[:no_prompts], all=True, std=True)
old_score = score_metric(model, ioi_dataset[:no_prompts])
model.reset_hooks()
model, _ = do_circuit_extraction(
    mlps_to_remove=None,
    heads_to_keep=heads_to_keep,
    mlps_to_keep={},
    no_prompts=no_prompts,
    model=model,
    ioi_dataset=ioi_dataset[:no_prompts],
)

ldiff, std = logit_diff(model, ioi_dataset[:no_prompts], std=True, all=True)
score = score_metric(model, ioi_dataset[:no_prompts])

# %%
print(f"Logit difference = {ldiff.mean().item()} +/- {std}. score={score.item()}") 
print(f"Original logit_diff = {old_ld.mean()} +/- {old_std}. score={old_score}")

df = pd.DataFrame({"Logit difference":ldiff.cpu(), 
"Random (for separation)":np.random.random(len(ldiff)),
"beg":[prompt["text"][:10] for prompt in ioi_prompts[:no_prompts]], 
"sentence": [prompt["text"] for prompt in ioi_prompts[:no_prompts]],
"#tokens before first name": [prompt["text"].count("Then") for prompt in ioi_prompts[:no_prompts]],
"template": ioi_dataset[:no_prompts].templates_by_prompt,
"misc": [ (str(prompt["text"].count("Then")) +str(ioi_dataset[:no_prompts].templates_by_prompt[i])) for (i,prompt) in enumerate(ioi_prompts[:no_prompts])] })
#[ prompt["text"].count(prompt["IO"]) for (i,prompt) in enumerate(ioi_prompts)] })

# TODO figure out how to make the global IOI dataset work

px.scatter(df, x="Logit difference", y="Random (for separation)", hover_data=["sentence", "template"],text="beg", color="misc", title=f"Prompt type = {ioi_dataset.prompt_type}")

# %%
show_tokens("After   Rebecca and James went to the grocery store. Rebecca gave a basketball to James")

# %%
for key in ioi_dataset.word_idx:
    print(key, ioi_dataset.word_idx[key][8])

# %% [markdown]
# # Global Patching

# %%

def do_global_patching(
    source_mlps_to_patch=None,
    source_mlps_to_keep=None,
    target_mlps_to_patch=None,
    target_mlps_to_keep=None,

    source_heads_to_keep=None,
    source_heads_to_patch=None,

    target_heads_to_keep=None,
    target_heads_to_patch=None,
    no_prompts=0,
    source_ioi_dataset=None,
    target_ioi_dataset=None,
    model=None,
):
    """
    if `ablate` then ablate all `heads` and `mlps`
        and keep everything else same
    otherwise, ablate everything else
        and keep `heads` and `mlps` the same
    """
    # check if we are either in keep XOR remove move from the args

    patching, heads, mlps = get_circuit_replacement_hook(
        target_heads_to_patch, #head
        target_mlps_to_patch,
        target_heads_to_keep,
        target_mlps_to_keep,
        source_heads_to_patch, #head2
        source_mlps_to_patch,
        source_heads_to_keep,
        source_mlps_to_keep,
        no_prompts,
        target_ioi_dataset,
        model,
    )

    config = PatchingConfig(
        patch_fn=patching,
        source_dataset=source_ioi_dataset.text_prompts[:no_prompts],  # TODO nb of prompts useless ?
        target_dataset=target_ioi_dataset.text_prompts[:no_prompts],
        target_module="attn_head",
        head_circuit="result",
        verbose=True,
        cache_act=True,
    )
    ptch = EasyPatching(
        model,
        config,
        metric,
    )
    model.reset_hooks()

    for layer, head in heads.keys():
        model.add_hook(*ptch.get_hook(layer, head))
    for layer in mlps.keys():
        model.add_hook(*ptch.get_hook(layer, head=None, target_module="mlp"))

    return model, ptch

# %%
N=100
no_prompts = N
target_ioi_dataset = IOIDataset(prompt_type="mixed", N=N, symmetric=True, prefixes=None)
source_ioi_dataset = target_ioi_dataset.gen_flipped_prompts("IO")

# %%
target_ioi_dataset.text_prompts[:3]

# %%
source_ioi_dataset.text_prompts[:3]

# %%

source_heads_to_keep, source_mlps_to_keep = get_heads_circuit(
    source_ioi_dataset, no_prompts, calib_head=False, mlp0=True
)
target_heads_to_keep, target_mlps_to_keep = get_heads_circuit(
    target_ioi_dataset, no_prompts, calib_head=False, mlp0=True
)

# %%
model.reset_hooks()
logit_diff(model, ioi_dataset[:no_prompts], all=False, std=True)

# %%
np.round(5.473965938, 4)

# %%
print_top_k(model, target_ioi_dataset, K=5)

# %%


# %%
K = 1
model.reset_hooks()
old_ld, old_std = logit_diff(
    model, target_ioi_dataset[:no_prompts], target_dataset=target_ioi_dataset, all=True, std=True
)
model.reset_hooks()
old_score = score_metric(model, target_ioi_dataset[:no_prompts], target_dataset=target_ioi_dataset, k=K)
model.reset_hooks()
old_ld_source, old_std_source = logit_diff(
    model, target_ioi_dataset[:no_prompts], target_dataset=source_ioi_dataset, all=True, std=True
)
model.reset_hooks()
old_score_source = score_metric(model, target_ioi_dataset[:no_prompts], target_dataset=source_ioi_dataset, k=K)
model.reset_hooks()
model, _ = do_global_patching(
    source_mlps_to_patch=source_mlps_to_keep,
    source_mlps_to_keep=None,
    target_mlps_to_patch=target_mlps_to_keep,
    target_mlps_to_keep=None,
    source_heads_to_keep=None,
    source_heads_to_patch=source_heads_to_keep,
    target_heads_to_keep=None,
    target_heads_to_patch=target_heads_to_keep,
    no_prompts=N,
    model=model,
    source_ioi_dataset=source_ioi_dataset[:no_prompts],
    target_ioi_dataset=target_ioi_dataset[:no_prompts],
)

ldiff_target, std_ldiff_target = logit_diff(
    model, target_ioi_dataset[:no_prompts], target_dataset=target_ioi_dataset, std=True, all=True
)
score_target = score_metric(model, target_ioi_dataset[:no_prompts], target_dataset=target_ioi_dataset, k=K)
ldiff_source, std_ldiff_source = logit_diff(
    model, target_ioi_dataset[:no_prompts], target_dataset=source_ioi_dataset, std=True, all=True
)
score_source = score_metric(model, target_ioi_dataset[:no_prompts], target_dataset=source_ioi_dataset, k=K)
# %%
print(f"Original logif_diff TARGET DATASET (TARGET, no patching)=  {old_ld.mean()} +/- {old_std}. Score {old_score}")
print(
    f"Original logif_diff TARGET DATASET  (SOURCE, no patching)=  {old_ld_source.mean()} +/- {old_std_source}. Score {old_score_source}"
)
print(f"Logit_diff TARGET (*AFTER* patching)=  {ldiff_target.mean()} +/- {std_ldiff_target}. Score {score_target}")
print(f"Logit_diff SOURCE (*AFTER* patching)=  {ldiff_source.mean()} +/- {std_ldiff_source}. Score {score_source}")
df = pd.DataFrame(
    {
        "x": ldiff_source.cpu(),
        "y": np.random.random(len(ldiff_source)),
        "beg": [prompt["text"][:10] for prompt in ioi_prompts],
        "sentence": [prompt["text"] for prompt in ioi_prompts],
        "#tokens before first name": [prompt["text"].count("Then") for prompt in ioi_prompts],
        "template": ioi_dataset.templates_by_prompt,
        "misc": [
            (str(prompt["text"].count("Then")) + str(ioi_dataset.templates_by_prompt[i]))
            for (i, prompt) in enumerate(ioi_prompts)
        ],
    }
)
# [ prompt["text"].count(prompt["IO"]) for (i,prompt) in enumerate(ioi_prompts)] })
px.scatter(
    df, x="x", y="y", hover_data=["sentence", "template"], text="beg", color="misc", title=ioi_dataset.prompt_type
)
# %%
"ssodiqsddifusgidfuisd".index("s")