# %%
from functools import *
from collections import OrderedDict
import graphviz
import itertools
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
import spacy

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
pio.renderers.default = "vscode"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

# %%
from easy_transformer.hook_points import HookedRootModule, HookPoint
from easy_transformer.EasyTransformer import EasyTransformer
from easy_transformer.experiments import ExperimentMetric, AblationConfig, EasyAblation, EasyPatching, PatchingConfig

from easy_transformer_utils import (
    show_tokens, 
    sample_next_token,
    get_topk_completions, 
    show_pp,
    show_attention_patterns,
    get_OV_circuit_output,
    get_bigram_freq_output
    )

from utils_circuit_discovery import HypothesisTree, show_graph

# %% Load model

model_name = "gpt2"  # @param ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'facebook/opt-125m', 'facebook/opt-1.3b', 'facebook/opt-2.7b', 'facebook/opt-6.7b', 'facebook/opt-13b', 'facebook/opt-30b', 'facebook/opt-66b', 'EleutherAI/gpt-neo-125M', 'EleutherAI/gpt-neo-1.3B', 'EleutherAI/gpt-neo-2.7B', 'EleutherAI/gpt-j-6B', 'EleutherAI/gpt-neox-20b']
model = EasyTransformer.from_pretrained(model_name) #, use_attn_result=True)
if torch.cuda.is_available():
    model.to("cuda")
model.set_use_attn_result(True)
#print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# %% Probe the acronym generation behavior
input = "On Thursday afternoon, the British Vision Ventures ("
input = "This sentence supplies a different open paren ("
#input = "In a statement released by the Big Government Agency ("
show_tokens(model, input)
get_topk_completions(model, input, k=10, device=device)

def generate_completion(model: EasyTransformer, input: str, max_len: int = 3, device=device):
    assert isinstance(input, str)
    l = 0
    log_prob = 0
    while l < max_len:
        input_tokens = model.tokenizer.encode(input)
        logits = sample_next_token(model, torch.LongTensor(input_tokens).to(device))
        top_value, top_index = torch.topk(logits, k=1)
        log_softmax = nn.LogSoftmax(dim=-1)
        log_prob += log_softmax(logits)[top_index].item()
        #print(log_prob)
        generated = model.tokenizer.decode(top_index)
        l += len(generated)
        input += generated
    return input, log_prob

print("Generating a completion...")
generate_completion(model, input, max_len=1)

# %% Find the two- and three-letter tokens

uppers = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
two_letter_single_token = []
for l1, l2 in itertools.product(uppers, uppers):
    acro = l1 + l2
    if len(model.to_tokens(acro, prepend_bos=False)[0]) == 1:
        two_letter_single_token.append(acro)

three_letter_single_token = []
all_individual = []

for l1, l2, l3 in itertools.product(uppers, uppers, uppers):
    acro = l1 + l2 + l3
    if len(model.to_tokens(acro, prepend_bos=False)[0]) == 1:
        three_letter_single_token.append(acro)
    elif acro[:2] not in two_letter_single_token and acro[1:] not in two_letter_single_token:
        all_individual.append(acro)

print(f"{len(two_letter_single_token)} two-letter acronyms are their own tokens, which is {len(two_letter_single_token)/(26*26)} of all {26 * 26} possible two-letter acronyms.")

print(f"{len(three_letter_single_token)} three-letter acronyms are their own tokens, which is {len(three_letter_single_token)/(26*26*26)} of all {26*26*26} possible three-letter acronyms.")

print(f"{len(all_individual)} three-letter acronyms don't contain any two-letter substring that are one token, which is {len(all_individual)/(26*26*26)} of all {26*26*26} possible three-letter acronyms.")


# %% What are the tokens that are proper nouns?

vocab_size = model.tokenizer.vocab_size
proper_noun_ish = []
for i in range(vocab_size):
    token = model.tokenizer.decode(i)
    if len(token) < 6: # filter out ones that are too short
        continue
    if token[0] != ' ':
        continue
    if token[1] not in uppers:
        continue
    if token[2] in uppers:
        continue
    proper_noun_ish.append(token)

nlp = spacy.load("en_core_web_sm")
text = ''.join(proper_noun_ish)
doc = nlp(text)
proper_noun_ish = []
for token in doc:
    if token.pos_ == 'PROPN':
        proper_noun_ish.append(token.text)

# %%
# Make Fake Orgs dataset
# Attempt: make word combos that mostly make sense:
# adj + noun + noun(organization/entity)
# [] of [] []? Board of Education, Journal of High Energy Physics, Department of Motor Vehicles, University of ..., Republic of ...

with open("fake_orgs.json", "rb") as f:
    data = json.load(f)

adjs = data["adjs"]
nouns = data['nouns']
orgs = data["orgs"]
prompts = data["prompts"]

def differ_only_at_first_token(acro1: str, acro2: str) -> bool:
    tokens1 = model.to_tokens(acro1, prepend_bos=False).squeeze(0)
    tokens2 = model.to_tokens(acro2, prepend_bos=False).squeeze(0)
    if len(tokens1) != len(tokens2) or len(tokens1) == 1:
        return False
    if tokens1[0] != tokens2[0] and (tokens1[1:] == tokens2[1:]).prod():
        return True
    return False

def get_acro_that_differs_at_first(acro: str):
    tokens = model.to_tokens(acro, prepend_bos=False).squeeze(0)
    if len(tokens) == 1: # cannot be done if the whole acronym is one token
        return None
    first_token_str = model.tokenizer.decode(tokens[0])
    rest_str = model.tokenizer.decode(tokens[1:])
    while True:
        if len(first_token_str) == 1:
            idx = np.random.choice(26)
            X = uppers[idx]
        else:
            idx = np.random.choice(len(two_letter_single_token))
            X = two_letter_single_token[idx]
        new_acro = X + rest_str
        assert len(new_acro) == 3
        if differ_only_at_first_token(acro, new_acro):
            return new_acro

def differ_only_at_second_token(acro1: str, acro2: str) -> bool:
    tokens1 = model.to_tokens(acro1, prepend_bos=False).squeeze(0)
    tokens2 = model.to_tokens(acro2, prepend_bos=False).squeeze(0)
    if len(tokens1) != len(tokens2) or len(tokens1) == 1:
        return False
    if tokens1[1] != tokens2[1] and tokens1[0] == tokens2[0]:
        if len(tokens1) == 2:
            return True
        else:
            if tokens1[2] == tokens2[2]:
                return True
    return False

def get_acro_that_differs_at_second(acro: str):
    tokens = model.to_tokens(acro, prepend_bos=False).squeeze(0)
    if len(tokens) == 1: # cannot be done if the whole acronym is one token
        return None
    first_token_str = model.tokenizer.decode(tokens[0])
    second_token_str = model.tokenizer.decode(tokens[1])
    if len(tokens) > 2:
        third_token_str = model.tokenizer.decode(tokens[2])
    #while True:
    if len(second_token_str) == 1:
        idx = np.random.choice(26)
        X = uppers[idx]
    else:
        idx = np.random.choice(len(two_letter_single_token))
        X = two_letter_single_token[idx]
    new_acro = first_token_str + X
    if len(tokens) > 2:
        new_acro += third_token_str
    assert len(new_acro) == 3
    #print(acro, new_acro)
    if differ_only_at_second_token(acro, new_acro):
        return new_acro 
    else:
        return None

def get_acro_that_differs_at_pos(acro: str, pos: int):
    """ This assumes X|X|X tokenization """
    assert pos in [0, 1 ,2]
    tokens = model.to_tokens(acro, prepend_bos=False).squeeze(0)
    assert len(tokens) == 3
    idx = np.random.choice(26)
    X = uppers[idx]
    new_acro = ""
    for i in range(3):
        if i == pos:
            new_acro += X
        else:
            new_acro += acro[i]
    new_tokens = model.to_tokens(new_acro, prepend_bos=False).squeeze(0)
    if len(new_tokens) == 3 and new_acro != acro:
        return new_acro
    else:
        return None

def get_phrase_with_acro(acro: str, words1: list[str], words2: list[str], words3: list[str]):
    A, B, C = acro
    words1c = [word for word in words1 if word[0] == A]
    words2c = [word for word in words2 if word[0] == B]
    words3c = [word for word in words3 if word[0] == C]
    if len(words1c) == 0 or len(words2c) == 0 or len(words3c) == 0:
        return None
    word1 = np.random.choice(words1c)
    word2 = np.random.choice(words2c)
    word3 = np.random.choice(words3c)
    return word1 + ' ' + word2 + ' ' + word3

def make_dataset(n: int, words1: list[str], words2: list[str], words3: list[str], prompts: list[str], mode="default"):
    if mode not in ["default", "first", "second", "third", "111", "12", "21"]:
        raise NotImplementedError
    dataset = []
    for i in tqdm(range(n)):
        acro2 = None
        phrase2 = None
        while phrase2 is None: # to make sure we don't get invalid acros (single-token ones)
            words1c = np.random.choice(words1)
            words2c = np.random.choice(words2)
            words3c = np.random.choice(words3)
            promptsc = np.random.choice(prompts)
            promptsc = "On Thursday morning, the {} ({})"
            phrase = words1c + ' ' + words2c + ' ' + words3c
            acro = words1c[0] + words2c[0] + words3c[0]
            acro_len = len(model.to_tokens(acro, prepend_bos=False)[0])
            if mode == "default":
                acro2 = acro
            if mode == "first":
                if acro_len == 3:
                    acro2 = get_acro_that_differs_at_pos(acro, 0)
            if mode == "second":
                if acro_len == 3:
                    acro2 = get_acro_that_differs_at_pos(acro, 1)
            if mode == "third":
                if acro_len == 3:
                    acro2 = get_acro_that_differs_at_pos(acro, 2)
            if mode == "111":
                if acro_len == 3:
                    acro2 = acro
            if mode == "12":
                if acro_len == 2 and len(model.to_tokens(acro[1:], prepend_bos=False)[0]) == 1:
                    acro2 = acro
            if mode == "21":
                if acro_len == 2 and len(model.to_tokens(acro[:2], prepend_bos=False)[0]) == 1:
                    acro2 = acro
            if acro2 is None:
                continue
            phrase2 = get_phrase_with_acro(acro2, words1, words2, words3)
        if mode == "default" or mode == "111" or mode == "12" or mode == "21":
            full_prompt = promptsc.format(phrase, acro)
            dataset.append({"full_prompt": full_prompt, "phrase": phrase, "acro": acro})
        if mode == "first" or mode == "second" or mode == "third":
            prompt = promptsc.format(phrase, acro)
            prompt2 = promptsc.format(phrase2, acro2)
            dataset.append(({"full_prompt": prompt, 
                             "phrase": phrase, 
                             "acro": acro},
                             {"full_prompt": prompt2, 
                             "phrase": phrase2, 
                             "acro": acro2}))
    return dataset

# %%

fake_orgs_111 = make_dataset(10, adjs, nouns, orgs, prompts, mode="111")

print("Example fake organizations:")
print([data["phrase"] for data in fake_orgs_111])

# %% FakeOrgsDataset class

# Interp experiments
# 1. Which heads change the log prob most when ablated?
# 2. Various patching: acronym transfer and word swap
# 3. Which heads write most strongly in the direction of the right answer?

def get_acro_pos(text_prompt: str):
    """ Assumes that the text prompt contains substring '([acro])'.
        Returns the token indices from the first token in acro to 
        the close paren. """
    tokens = list(model.to_tokens(text_prompt).squeeze())
    open_paren = model.tokenizer.encode(" (")[0]
    close_paren = model.tokenizer.encode(")")[0]
    open_idx = tokens.index(open_paren)
    close_idx = tokens.index(close_paren)
    return torch.arange(open_idx, close_idx)

def get_token_pos(sentence: str, token_word: str, prepend_bos: bool = True):
    assert token_word in sentence
    token = model.to_tokens(token_word, prepend_bos=False)[0]
    assert len(token) == 1
    token = token.item()
    sentence_tokens = list(model.to_tokens(sentence, prepend_bos=prepend_bos)[0])
    return sentence_tokens.index(token)

class FakeOrgsDataset():
    def __init__(self, N: int, dataset: list[dict] = None, tokenizer=None, mode="default"):
        self.N = N
        self.open_paren = tokenizer.encode(" (")[0]
        self.close_paren = tokenizer.encode(")")[0]

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = tokenizer

        if mode not in ["default", "first", "second", "third", "111", "12", "21"]:
            raise NotImplementedError

        if dataset is None:
            with open("fake_orgs.json", "rb") as f:
                data = json.load(f)
            adjs = data["adjs"]
            nouns = data['nouns']
            orgs = data["orgs"]
            prompts = data["prompts"]
            self.dataset = make_dataset(N, adjs, nouns, orgs, prompts, mode)
        else:
            self.dataset = dataset

        if mode == "default" or mode == "111" or mode == "12" or mode == "21":
            self.text_prompts = [data["full_prompt"] for data in self.dataset]
            self.toks = model.to_tokens([data["full_prompt"] for data in self.dataset])
            self.acros = [data["acro"] for data in self.dataset]
            self.acro_pos = [get_acro_pos(prompt) for prompt in self.text_prompts]
            self.max_len = max([ len(self.tokenizer(prompt).input_ids) for prompt in self.text_prompts ]) + 1

        if mode == "first" or mode == "second" or mode == "third":
            self.text_prompts = ([data[0]["full_prompt"] for data in self.dataset],
                                 [data[1]["full_prompt"] for data in self.dataset])
            self.toks = (model.to_tokens([data[0]["full_prompt"] for data in self.dataset]), 
                         model.to_tokens([data[1]["full_prompt"] for data in self.dataset]))
            # self.toks = (torch.tensor(
            #     self.tokenizer([data[0]["full_prompt"] for data in self.dataset], padding=True).input_ids, dtype=torch.int),
            #              torch.tensor(
            #     self.tokenizer([data[1]["full_prompt"] for data in self.dataset], padding=True).input_ids, dtype=torch.int))
            self.acros = ([data[0]["acro"] for data in self.dataset],
                          [data[1]["acro"] for data in self.dataset])
            self.acro_pos = ([get_acro_pos(prompt) for prompt in self.text_prompts[0]],
                             [get_acro_pos(prompt) for prompt in self.text_prompts[1]])
            self.max_len = max([ len(self.tokenizer(prompt[0]).input_ids) for prompt in self.text_prompts ]) + 1

# %% acro_log_probs and first_letter_log_probs
# This gives the same answer as evaluate_dataset_log_prob, but much faster

def acro_log_probs(model: EasyTransformer, text_prompts: list[str], dataset: FakeOrgsDataset):
    with torch.inference_mode():
        tot_log_probs = []
        N = len(text_prompts)
        logits = model(text_prompts)
        log_softmax = nn.LogSoftmax(dim=-1)
        log_probs = log_softmax(logits)
        for i in range(N):
            pos = dataset.acro_pos[i]
            acro_log_probs = log_probs[i, pos, dataset.toks[i][pos+1].long()]
            tot_log_prob = torch.log(torch.exp(acro_log_probs).prod()).item()
            tot_log_probs.append(tot_log_prob)
    return np.array(tot_log_probs)

def first_letter_log_probs(model: EasyTransformer, text_prompts: list[str], dataset: FakeOrgsDataset):
    with torch.inference_mode():
        tot_log_probs = []
        N = len(text_prompts)
        logits = model(text_prompts)
        log_softmax = nn.LogSoftmax(dim=-1)
        log_probs = log_softmax(logits)
        for i in range(N):
            pos = dataset.acro_pos[i][0]
            acro_log_probs = log_probs[i, pos, dataset.toks[i][pos+1].long()]
            tot_log_prob = torch.log(torch.exp(acro_log_probs).prod()).item()
            tot_log_probs.append(tot_log_prob)
    return np.array(tot_log_probs)

def nth_letter_log_probs(model: EasyTransformer, dataset: FakeOrgsDataset, n: int):
    ''' 0-indexing e.g. n=0 for the first letter in the acronym'''
    with torch.inference_mode():
        tot_log_probs = []
        N = dataset.N
        logits = model(dataset.text_prompts)
        log_softmax = nn.LogSoftmax(dim=-1)
        log_probs = log_softmax(logits)
        for i in range(N):
            pos = dataset.acro_pos[i][n]
            acro_log_probs = log_probs[i, pos, dataset.toks[i][pos+1].long()]
            tot_log_prob = torch.log(torch.exp(acro_log_probs).prod()).item()
            tot_log_probs.append(tot_log_prob)
    return np.array(tot_log_probs)

# %% Make the datasets
fake_orgs_dataset_1 = FakeOrgsDataset(50, tokenizer=model.tokenizer, mode="111")
fake_orgs_dataset_2 = FakeOrgsDataset(50, tokenizer=model.tokenizer, mode="111")

# %% Load OpenWebText for mean ablation

# webtext = load_dataset("stas/openwebtext-10k")
# owb_seqs = ["".join(show_tokens(model, webtext['train']['text'][i][:2000], return_list=True)[1:fake_orgs_dataset_1.max_len + 2]) for i in range(100)]

# %%

positions = OrderedDict()
positions['THE'] = torch.tensor([5 for i in range(fake_orgs_dataset_1.N)])
positions['W1'] = torch.tensor([6 for i in range(fake_orgs_dataset_1.N)])
positions['W2'] = torch.tensor([7 for i in range(fake_orgs_dataset_1.N)])
positions['W3'] = torch.tensor([8 for i in range(fake_orgs_dataset_1.N)])
positions['OP'] = torch.tensor([9 for i in range(fake_orgs_dataset_1.N)])
positions['L1'] = torch.tensor([10 for i in range(fake_orgs_dataset_1.N)])
positions['L2'] = torch.tensor([11 for i in range(fake_orgs_dataset_1.N)])


def first_letter_metric(model: EasyTransformer, dataset: FakeOrgsDataset):
    log_probs = nth_letter_log_probs(model, dataset, 0)
    return log_probs.mean()

def second_letter_metric(model: EasyTransformer, dataset: FakeOrgsDataset):
    log_probs = nth_letter_log_probs(model, dataset, 1)
    return log_probs.mean()

def third_letter_metric(model: EasyTransformer, dataset: FakeOrgsDataset):
    log_probs = nth_letter_log_probs(model, dataset, 2)
    return log_probs.mean()

h = HypothesisTree(
    model, 
    metric=third_letter_metric, 
    dataset=fake_orgs_dataset_1,
    orig_data=fake_orgs_dataset_1.toks.long(), 
    new_data=fake_orgs_dataset_2.toks.long(), 
    threshold=0.2,
    possible_positions=positions,
    use_caching=True
)

# %%
h.eval(verbose=True, show_graphics=True)
#h.show(save_file='acronym_graph_medium.png')
while h.current_node is not None:
    h.eval(auto_threshold=True, verbose=True, show_graphics=True)
    #h.show(save_file='acronym_graph_medium.png')
    with open('acronym_graph_3rd_letter_small.pkl', 'wb') as f:
        pickle.dump(h, f, pickle.HIGHEST_PROTOCOL)

# big graph, run in terminal
# h.eval(verbose=True, show_graphics=False)
# #h.show(save_file='acronym_graph_medium.png')
# while h.current_node is not None:
#     h.eval(threshold=0.01, verbose=True, show_graphics=False)
#     #h.show(save_file='acronym_graph_medium.png')
#     with open('acronym_graph_2nd_letter_big.pkl', 'wb') as f:
#         pickle.dump(h, f, pickle.HIGHEST_PROTOCOL)
# %%
with open('acronym_graph_2nd_letter_big.pkl', 'rb') as f:
    h2big = pickle.load(f)

# %%
show_graph(h2small)
# %%
