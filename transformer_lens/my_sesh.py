#%%

# TODO(conmy): autoreload?

import gc
from functools import partial

# pyright: reportMissingImports=false
from typing import *

import datasets
import einops
import jax
import jaxtyping as jt
import sae_vis
import torch
from IPython.display import HTML, display
from safetensors import safe_open

import transformer_lens

torch.set_grad_enabled(False)

#%%

anger_words = ["Fury",
"aggravation",
"aggravationed",
"aggravationing",
"aggravations",
"aggravator",
"aggravatorred",
"aggravatorring",
"aggravators",
"ailed",
"ailing",
"ails",
"allium_tricoccummed",
"allium_tricoccumming",
"allium_tricoccums",
"anger",
"angered",
"angering",
"angers",
"angried",
"angries",
"angriness",
"angrinessed",
"angrinessing",
"angry",
"angrying",
"anguish",
"anguished",
"anguishes",
"anguishing",
"annoy",
"annoyance",
"annoyanced",
"annoyances",
"annoyancing",
"annoyed",
"annoying",
"annoys",
"bawl_out",
"bawl_outed",
"bawl_outing",
"bawl_outs",
"bedevil",
"bedevilled",
"bedevilling",
"bedevils",
"berate",
"berated",
"berates",
"berating",
"bilk",
"bilked",
"bilking",
"bilks",
"botheration",
"botherationed",
"botherationing",
"botherations",
"bothering",
"bothersome",
"bothersomed",
"bothersomes",
"bothersoming",
"cacoethed",
"cacoethes",
"cacoething",
"call_down",
"call_downed",
"call_downing",
"call_downs",
"call_on_the_carpet",
"call_on_the_carpets",
"call_on_the_carpetted",
"call_on_the_carpetting",
"chafe",
"chafed",
"chafes",
"chafing",
"chew_out",
"chew_outed",
"chew_outing",
"chew_outs",
"chew_up",
"chew_upped",
"chew_upping",
"chew_ups",
"chide",
"chided",
"chides",
"chiding",
"choler",
"cholerred",
"cholerring",
"cholers",
"cod",
"codded",
"codding",
"cods",
"concern",
"concerned",
"concerning",
"concerns",
"craze",
"crazed",
"crazes",
"craziness",
"crazinessed",
"crazinessing",
"crazing",
"cross",
"cross_thwart",
"cross_thwarted",
"cross_thwarting",
"cross_thwarts",
"crossed",
"crosses",
"crossing",
"crossness",
"crossnessed",
"crossnessing",
"crucified",
"crucifies",
"crucify",
"crucifying",
"cult",
"culted",
"culting",
"cults",
"cultued",
"cultuing",
"cultus",
"daunted",
"daunting",
"daunts",
"delirium",
"deliriumed",
"deliriuming",
"deliriums",
"deranged",
"deranges",
"deranging",
"despised",
"despises",
"despising",
"detest",
"detested",
"detesting",
"detests",
"devil",
"deviled",
"deviling",
"devils",
"disappointed",
"disappointing",
"disappoints",
"discomfited",
"discomfiting",
"discomfits",
"discomfort",
"discomforted",
"discomforting",
"discomforts",
"discommode",
"discommoded",
"discommodes",
"discommoding",
"disoblige",
"disobliged",
"disobliges",
"ferocitied",
"ferocities",
"ferocity",
"ferocitying",
"fierceness",
"fiercenessed",
"fiercenessing",
"foil",
"foiled",
"foiling",
"foils",
"follied",
"follies",
"folly",
"follying",
"foolishness",
"foolishnessed",
"foolishnessing",
"frustrate",
"frustrated",
"frustrates",
"frustrating",
"frustration",
"frustrationed",
"frustrationing",
"frustrations",
"frustrative",
"frustratived",
"frustratives",
"frustrativing",
"furied",
"furies",
"furioued",
"furiouing",
"furious",
"furiousness",
"furiousnessed",
"furiousnessing",
"furor",
"furore",
"furored",
"furores",
"furoring",
"furorred",
"furorring",
"furors",
"fury",
"furying",
"hate",
"hated",
"hates",
"hating",
"hatred",
"hatres",
"hatring",
"hats",
"ire",
"ired",
"ires",
"iring",
"irritabilities",
"irritability",
"irritabilitying",
"irritate",
"irritated",
"irritates",
"irritating",
"irritation",
"irritationed",
"irritationing",
"irritations",
"kill",
"killed",
"killing",
"kills",
"lambast",
"lambaste",
"lambasted",
"lambasting",
"lambasts",
"madden",
"mad",
"maddened",
"maddening",
"maddens",
"madness",
"madnessed",
"madnessing",
"mania",
"maniaed",
"maniaing",
"manias",
"manic",
"manices",
"manicked",
"manicking",
"miffed",
"miffing",
"miffs",
"nuisance",
"nuisanced",
"nuisances",
"nuisancing",
"offended",
"offending",
"offends",
"overcame",
"overcome",
"overcomes",
"overcoming",
"pain",
"pain_in_the_ass",
"pain_in_the_assed",
"pain_in_the_assing",
"pain_in_the_neck",
"pain_in_the_necked",
"pain_in_the_necking",
"pain_in_the_necks",
"pain_sensation",
"pain_sensationed",
"pain_sensationing",
"pain_sensations",
"pained",
"painful_sensation",
"painful_sensationed",
"painful_sensationing",
"painful_sensations",
"painfulness",
"painfulnessed",
"painfulnessing",
"paining",
"pains",
"pissed",
"pissed_off",
"pissed_offed",
"pissed_offing",
"pissed_offs",
"pisses",
"pissing",
"rage",
"raged",
"rages",
"ragged",
"ragging",
"raging",
"rags",
"remonstrate",
"remonstrated",
"remonstrates",
"remonstrating",
"reprimand",
"reprimanded",
"reprimanding",
"reprimands",
"reproof",
"reproofed",
"reproofing",
"reproofs",
"rile",
"riled",
"riles",
"riling",
"roiled",
"roiling",
"roils",
"scold",
"scolded",
"scolding",
"scolds",
"scorned",
"scorning",
"scorns",
"soreness",
"sorenessed",
"sorenessing",
"temper",
"tempered",
"tempering",
"tempers",
"tempest",
"tempested",
"tempesting",
"tempests",
"tempestuoued",
"tempestuouing",
"tempestuous",
"torment",
"tormented",
"tormenting",
"torments",
"transparencied",
"transparencies",
"transparency",
"transparencying",
"trouble_oneself",
"trouble_oneselfed",
"trouble_oneselfing",
"trouble_oneselfs",
"trounce",
"trounced",
"trounces",
"trouncing",
"twit",
"twits",
"twitted",
"twitting",
"vehemence",
"vehemenced",
"vehemences",
"vehemencing",
"vex",
"vexation",
"vexationed",
"vexationing",
"vexations",
"vexatioued",
"vexatiouing",
"vexatious",
"vexed",
"vexes",
"vexing",
"violence",
"violenced",
"violences",
"violencing",
"violent_storm",
"violent_stormed",
"violent_storming",
"violent_storms",
"wrath",
"wrathed",
"wrathing",
"wraths",
"arse",
"arsed",
"arses",
"arsing",
"ass",
"asshole",
"assholed",
"assholes",
"assholing",
"bastard",
"bastarded",
"bastarding",
"bastards",
"bitch",
"bitched",
"bitches",
"bitching",
"cock",
"cocked",
"cocking",
"cocks",
"cocksucker",
"cocksuckerred",
"cocksuckerring",
"cocksuckers",
"cunt",
"cunted",
"cunting",
"cunts",
"dick",
"dicked",
"dickhead",
"dickheaded",
"dickheading",
"dickheads",
"dicking",
"dicks",
"fuck",
"fucked",
"fucking",
"fucks",
"idiot",
"idioting",
"idiots",
"imbecile",
"imbeciled",
"imbeciles",
"imbeciling",
"moron",
"moronned",
"moronning",
"morons",
"motherfucker",
"motherfuckerred",
"motherfuckerring",
"motherfuckers",
"piss",
"pissed",
"pisses",
"pissing",
"prat",
"prats",
"pratted",
"pratting",
"prick",
"pricks",
"shit",
"shits",
"shitting",
"stern",
"sterned",
"sterning",
"sterns",
"twat",
"twats"
]

#%%

# <h1> Get Stepan's SAE's weights </h1>
# (Thanks Stepan! See https://colab.research.google.com/drive/18Toz1BIK8MGv0afC5gIKZ3leIUU_RGLZ)

tensors = {}
with safe_open("gpt2-20.safetensors", framework="pt") as f:
    for k in f.keys():
        tensors[k] = f.get_tensor(k)
tensors["b_enc"] += tensors["b_dec"] @ tensors["W_enc"]
# Sigh why is this here TODO(conmy): raise issue in SAELens
scaling_factor = tensors.pop("scaling_factor")
# Fold in -- this was trained on too.
tensors["W_enc"] *= scaling_factor
tensors["b_enc"] *= scaling_factor

tensors = jax.tree.map(lambda x: x.to("cuda:0").to(torch.float32), tensors)

#%%

gpt2xl = transformer_lens.HookedSAETransformer.from_pretrained_no_processing("gpt2-xl")  # SAE was trained without TL's nice things 

#%%

# Sanity check that this SAE is behaving as expected.

sae_cfg = transformer_lens.HookedSAEConfig(
    d_sae=1600 * 32,                  # from "d_sae" field in your config
    d_in=1600,                        # from "d_in" field in your config
    hook_name="blocks.20.hook_resid_pre",  # from "hook_point" field in your config
    use_error_term=False,             # assuming False as it's not specified in the provided config
    dtype=torch.float32,              # from "dtype" field in your config
    seed=42,                          # from "seed" field in your config
    device="cuda:0"                     # from "device" variable
)

sae_vis_sae_cfg = sae_vis.model_fns.AutoEncoderConfig(
    d_in = 1600,
    d_hidden = 1600*32,
    # device = "cuda:0"
)

sae = transformer_lens.HookedSAE(sae_cfg)

#%%

# scaling_factor = torch.ones((sae_cfg.d_sae), dtype=torch.float32, device="cuda:0")
# tensors["scaling_factor"] = scaling_factor
sae.load_state_dict(tensors)

#%%

data = datasets.load_dataset("Elriggs/openwebtext-100k", streaming=False)
data = data["train"]

#%%

iter_data = iter(data)

#%%

text = next(iter_data)["text"]
tokens = gpt2xl.to_tokens(text)

# %%

# Test run
site = "blocks.20.hook_resid_pre"
logits, all_acts = gpt2xl.run_with_cache(
    tokens,
    names_filter = site,
)

acts = all_acts[site]
print(acts.shape)

#%%

def get_neglogprobs(logits, tokens):
    neglogprobs = -logits.log_softmax(dim=-1)[
        torch.arange(logits.shape[0])[:, None],
        torch.arange(logits.shape[1]-1)[None],
        tokens[:, 1:]
    ]
    return neglogprobs

#%%

neglogprobs = get_neglogprobs(logits, tokens)

# %%

neglogprobs.shape, neglogprobs.mean()

#%%

sae_acts_pre_hook_name = "blocks.20.hook_resid_pre.hook_sae_acts_pre"
sae_logits, sae_cache = gpt2xl.run_with_cache_with_saes(
    tokens,
    saes = [sae],
    names_filter=sae_acts_pre_hook_name
)
sae_acts_pre = sae_cache[sae_acts_pre_hook_name]
print(sae_acts_pre.shape)

#%%

# Print avg L0
((sae_acts_pre > 0).sum(dim=-1).float()).mean()
# 47.

#%%

# from sae vis demo

SEQ_LEN = 128

# Tokenize the data (using a utils function) and shuffle it
tokenized_data = transformer_lens.utils.tokenize_and_concatenate(data, gpt2xl.tokenizer, max_length=SEQ_LEN) # type: ignore
tokenized_data = tokenized_data.shuffle(42)

# Get the tokens as a tensor
all_tokens = tokenized_data["tokens"]
assert isinstance(all_tokens, torch.Tensor)

print(all_tokens.shape)

#%%

# ugh gross TODO(conmy): raise issues to standardise the SAEs used by various libraries

sae_vis_sae = sae_vis.model_fns.AutoEncoder(sae_vis_sae_cfg).to("cuda:0")
sae_vis_sae.load_state_dict(tensors)

#%%

def get_sae_vis_data(features):
    # Specify the hook point you're using, the features you're analyzing, and the batch size for gathering activations
    sae_vis_config = sae_vis.SaeVisConfig(
        hook_point = site,
        features = features,
        batch_size = 2048,
        verbose = True,
    )

    # Gather the feature data
    return sae_vis.SaeVisData.create(
        encoder = sae_vis_sae,
        # encoder_B = encoder_B,
        model = gpt2xl,
        tokens = all_tokens, # type: ignore
        cfg = sae_vis_config,
    )

feature_idx = 126
sae_vis_data = get_sae_vis_data([feature_idx])

#%%

import time


def save_sae_vis_html(sae_vis_data):
    # Save as HTML file & open in browser (or not, if in Colab)
    filenames = []
    for feature_idx in sae_vis_data.feature_data_dict.keys():
        filename = f"feature_vis_demo_{int(1000*time.time())}.html"
        sae_vis_data.save_feature_centric_vis(filename, feature_idx=feature_idx)
        filenames.append(filename)
    return filenames

filename = save_sae_vis_html(sae_vis_data)[0]

#%%

# TODO(conmy): check all this!
display(HTML(filename))

#%%

sae_neglogprobs = get_neglogprobs(
    sae_logits,
    tokens,
)

# %%

sae_neglogprobs.mean()

# %%

# Using the AF post

pos_prompt = "Anger"  # @param {"type": "string"}
neg_prompt = "Calm"  # @param {"type": "string"}

pos_tokens = gpt2xl.to_tokens(pos_prompt, prepend_bos=True)
neg_tokens = gpt2xl.to_tokens(neg_prompt, prepend_bos=True)
assert pos_tokens.shape == neg_tokens.shape, (pos_tokens.shape, "!=" , neg_tokens.shape)

gpt2xl.reset_hooks()
_, cache = gpt2xl.run_with_cache(
    pos_tokens,
    names_filter = site,
)
pos_vec = cache[site][0, :]

gpt2xl.reset_hooks()
_, neg_cache = gpt2xl.run_with_cache(
    neg_tokens,
    names_filter = site,
)

neg_vec = neg_cache[site][0, :]
anger_steering_vec = 20*(pos_vec - neg_vec)

def activation_generation_hook(
    clean_activation: jt.Float[torch.Tensor, "Batch Seq *Dim"],
    hook: Any,
    indices: slice,
    v: jt.Float[torch.Tensor, "SubSeq *Dim"],
    debug: bool = False,
) -> jt.Float[torch.Tensor, "Batch Seq Dim"]:
  """TransformerLens hook only impacting prompt not rollout."""

  if clean_activation.shape[1] == 1:
    # Doing autoregression. No injection
    return clean_activation

  if debug:
    print("NORM PRE ADD", clean_activation[:, indices].norm(dim=-1))

  clean_activation[:, indices] += v

  if debug:
    print("NORM POST ADD", clean_activation[:, indices].norm(dim=-1))

  return clean_activation

prompt = "I think you're"
torch.random.manual_seed(100)

tokens = gpt2xl.to_tokens(prompt)

def get_steered_completion(
    tokens,
    steering_vec,
    indices,
):
    with gpt2xl.hooks(
        fwd_hooks=[
            (site, partial(activation_generation_hook, v=steering_vec, indices=indices))
        ]
    ):
        output = gpt2xl.generate(
            tokens,
            max_new_tokens=30,  # Params in Turner blog post
            top_p=0.3,
            temperature=1.0,
            freq_penalty=1.0,
            return_type="tensor"
        )
    return gpt2xl.to_string(output)

output = get_steered_completion(tokens, steering_vec=anger_steering_vec, indices=slice(0, 3))

#%%

print(output)

#%%

def is_angry(prompt: str, verbose: bool=False):
    for word in anger_words:
      if word.lower() in prompt.lower():
        if verbose:
            print(word)
        return True
    return False

# %%

is_angry(output[0], True)

# %%

many_tokens = einops.repeat(
   tokens,
   "1 Seq -> Batch Seq",
   Batch=100,
)

#%% 

many_outputs = get_steered_completion(
   many_tokens,
   steering_vec=anger_steering_vec,
   indices = slice(0, 3)
)

#%%

many_angry = [is_angry(prompt, False) for prompt in many_outputs]
many_angry.count(True) / len(many_angry)

#%%

angry_acts = gpt2xl.run_with_cache_with_saes(
   "Anger",
   names_filter=sae_acts_pre_hook_name,
   saes = [sae],
)[1][sae_acts_pre_hook_name]


#%%

most_firing_features = sorted(
   enumerate(angry_acts[0, -1].tolist()), key=lambda x:-x[1]
)

#%%

top20_firing_features = [x for x, _ in most_firing_features[:20]]

#%%

# What are we going to do now?
#
# Are there several anger features (cool).
#
# Could also look into the Eiffel Tower is in Rome steering vector.

#%%

pos_prompt = "The Eiffel Tower is in Rome"  # @param {"type": "string"}
neg_prompt = "The Eiffel Tower is in France"  # @param {"type": "string"}

pos_tokens = gpt2xl.to_tokens(pos_prompt, prepend_bos=True)
neg_tokens = gpt2xl.to_tokens(neg_prompt, prepend_bos=True)
assert pos_tokens.shape == neg_tokens.shape, (pos_tokens.shape, "!=" , neg_tokens.shape)

gpt2xl.reset_hooks()
_, cache = gpt2xl.run_with_cache(
    pos_tokens,
    names_filter = site,
)
pos_vec = cache[site][0, :]

gpt2xl.reset_hooks()
_, neg_cache = gpt2xl.run_with_cache(
    neg_tokens,
    names_filter = site,
)

neg_vec = neg_cache[site][0, :]
rome_steering_vec = (20*(pos_vec - neg_vec))[-1:, :]
# I hope this works

tokens = "To see the eiffel tower, people flock to"

#%%

output = get_steered_completion(tokens, steering_vec=rome_steering_vec, indices=slice(pos_tokens.shape[-1]-1, pos_tokens.shape[-1]))

print(output)

#%%

top20_sae_vis_data = get_sae_vis_data(top20_firing_features)
top20_filenames = save_sae_vis_html(top20_sae_vis_data)

#%%

# with open(top20_filenames[1], "r") as f:
display(HTML(filename=top20_filenames[1]))

#%%

rome_sae_logits, rome_sae_cache = gpt2xl.run_with_cache_with_saes(
    "The Eiffel Tower is in Rome",
    saes = [sae],
    names_filter=sae_acts_pre_hook_name
)

# %%

rome_sae_acts = rome_sae_cache[sae_acts_pre_hook_name]

#%%

most_firing_features = sorted(
   enumerate(rome_sae_acts[0, -1].tolist()), key=lambda x:-x[1]
)

#%%

top20_firing_features = [x for x, _ in most_firing_features[:20]]

#%%

top20_anger_features = [x for x, _ in [(126, 79.46971130371094),
 (20811, 41.01353454589844),
 (409, 31.83086585998535),
 (4364, 24.74541664123535),
 (44188, 24.631940841674805),
 (22759, 21.70770835876465),
 (4524, 20.727632522583008),
 (12006, 17.99246597290039),
 (33085, 15.927478790283203),
 (25116, 13.03805160522461),
 (11977, 12.467061996459961),
 (10578, 12.298257827758789),
 (25473, 11.171257019042969),
 (34590, 10.714902877807617),
 (21255, 9.855013847351074),
 (4254, 9.812237739562988),
 (12346, 9.630706787109375),
 (6150, 7.890009880065918),
 (21981, 7.673370361328125),
 (3640, 7.164831161499023),
 (37635, 6.826920509338379),
 (34020, 6.79866886138916),
 (24068, 6.5817365646362305),
 (40292, 6.227567672729492),
 (43381, 6.135743141174316),
 (48475, 5.722399711608887),
 (14552, 5.131052017211914),
 (24478, 4.506843090057373),
 (33228, 4.407247543334961),
 (20528, 4.191730499267578)]]

#%%

unique_rome_features = [x for x in top20_firing_features if x not in top20_anger_features]

# %%

rome_top20_sae_vis_data = get_sae_vis_data(unique_rome_features)
top20_filenames = save_sae_vis_html(rome_top20_sae_vis_data)

#%%

sae_steering_vec = sae.state_dict()["W_dec"][1211] * 29 * 20

#%%

output = get_steered_completion(tokens, steering_vec=sae_steering_vec, indices=slice(pos_tokens.shape[-1]-1, pos_tokens.shape[-1]))

# %%

print(output)

#%%

