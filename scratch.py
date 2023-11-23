# %%

from IPython import get_ipython

ipython = get_ipython()
if ipython is not None:
    ipython.magic("%load_ext autoreload")
    ipython.magic("%autoreload 2")

# Sad, really annoying to have to remember this
# import os
# os.environ["TRANSFORMERS_CACHE"] = "/workspace/cache"

# %%

from collections import defaultdict

import einops
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from transformer_lens import HookedTransformer, utils

# %%

model_name = "gpt2"
tl_model = HookedTransformer.from_pretrained_no_processing(model_name)
hf_model = AutoModelForCausalLM.from_pretrained(model_name)
hf_model.eval()

# %%

# tl_model = tl_model.to(torch.float64)
# hf_model = hf_model.to(torch.float64)

# %%

string = "Hello, world!"
tokens = tl_model.to_tokens(string)
logits, cache = tl_model.run_with_cache(tokens, prepend_bos=False)

# %%


class ActivationCacher:
    def __init__(self):
        self.activations = defaultdict(list)

    def cache_activations(self, module, module_name):
        def hook(module, input, output):
            self.activations[module_name].append(output)

        return hook


# %%

# Create an ActivationCacher instance
activation_cacher = ActivationCacher()

# Register hooks for caching activations
for name, module in hf_model.named_modules():
    module.register_forward_hook(activation_cacher.cache_activations(module, name))

# %%

hf_logits = hf_model(tokens).logits

# %%

torch.testing.assert_close(logits, hf_logits, atol=1e-9, rtol=1e-9)

# %%

# MODE = "ln_f"

# if MODE == "resid_post":
#     raise NotImplementedError()
#     tl_activation_name = utils.get_act_name("resid_post", 11)
#     hf_activation_name = "transformer"  # oh lol not really

# elif MODE == "attn":
#     tl_activation_name = "blocks.{layer_idx}.hook_attn_out"
#     if "gpt2" in model_name:
#         hf_activation_name = "transformer.h.{layer_idx}.attn"
#     else:
#         raise ValueError(f"Add this please! {activation_cacher.activations.keys()=}")
# elif MODE == "pattern":
#     tl_activation_name = utils.get_act_name("pattern", "{layer_idx}")  # lol
#     if "gpt2" in model_name:
#         hf_activation_name = "transformer.h.{layer_idx}.attn.attn_dropout"
#     elif "pythia" in model_name:
#         hf_activation_name = "gpt_neox.layers.{layer_idx}.attention.attention_dropout"
#     else:
#         raise ValueError(f"Add this please! {activation_cacher.activations.keys()=}")
# elif MODE == "mlp":
#     tl_activation_name = utils.get_act_name("mlp_out", "{layer_idx}")
#     assert "gpt2" in model_name
#     hf_activation_name = (
#         "transformer.h.{layer_idx}.mlp.dropout"  # Can try .mlp.c_fc... etc
#     )
# elif MODE == "mlp_pre":
#     tl_activation_name = utils.get_act_name("mlp_pre", "{layer_idx}")
#     assert "gpt2" in model_name
#     hf_activation_name = "transformer.h.{layer_idx}.mlp.c_fc"
# elif MODE == "mlp_post":
#     tl_activation_name = utils.get_act_name("mlp_post", "{layer_idx}")
#     assert "gpt2" in model_name
#     hf_activation_name = "transformer.h.{layer_idx}.mlp.c_proj"
# elif MODE == "embed":
#     assert "gpt2" in model_name
#     tl_activation_name = utils.get_act_name("resid_pre", 0)
#     hf_activation_name = "transformer.drop"
# elif MODE == "ln":
#     tl_activation_name = "blocks.{layer_idx}.ln1.hook_normalized"
#     assert "gpt2" in model_name
#     hf_activation_name = "transformer.h.{layer_idx}.ln_1"
# elif MODE == "ln2":
#     tl_activation_name = "blocks.{layer_idx}.ln2.hook_normalized"
#     assert "gpt2" in model_name
#     hf_activation_name = "transformer.h.{layer_idx}.ln_2"
# elif MODE == "qkv":
#     tl_activation_name = utils.get_act_name("q", "{layer_idx}")
#     assert "gpt2" in model_name
#     hf_activation_name = "transformer.h.{layer_idx}.attn.c_attn"
# elif MODE == "ln_f":
#     tl_activation_name = "ln_final.hook_normalized"
#     assert "gpt2" in model_name
#     hf_activation_name = "transformer.ln_f"

# else:
#     raise ValueError(f"Add this please! {MODE=}")

# saved_preln = activation_cacher.activations["transformer.drop"][0]
# # hook_normalized

# for i in range(tl_model.cfg.n_layers):
#     print(i)
#     tl_act = cache[
#         tl_activation_name.format(layer_idx=i)
#         if "{layer_idx}" in tl_activation_name
#         else tl_activation_name
#     ]

#     hf_act = activation_cacher.activations[
#         hf_activation_name.format(layer_idx=i)
#         if "{layer_idx}" in hf_activation_name
#         else hf_activation_name
#     ][0]

#     if MODE in "qkv":
#         query, key, value = hf_act.split(tl_model.cfg.d_model, dim=2)
#         if MODE == "q":
#             hf_act = einops.rearrange(
#                 query, "b s (h d) -> b s h d", h=tl_model.cfg.n_heads
#             )
#         else:
#             raise NotImplementedError()
#     elif MODE == "attn":
#         hf_act = hf_act[0]
#     elif MODE == "resid_post":
#         hf_act = hf_act["last_hidden_state"]

#     try:  # Suppress
#         torch.testing.assert_close(
#             tl_act.to(torch.float32),
#             hf_act.to(torch.float32),
#             atol=1e-9,  # Wow, embed is super close
#             rtol=1e-9,  # Better, but still failing a lot!
#             # msg="Hello",
#         )
#     except Exception as e:
#         a = str(e)
#     else:
#         a = None

#     if a is not None:
#         print(a)
#         assert False

# else:
#     print("All good!")

# # %%

# torch.testing.assert_close(
#     # tl_model.W_Q[0, 0],
#     # hf_model.transformer.h[0]
#     # .attn.c_attn.weight.split(tl_model.cfg.d_model, dim=-1)[0]
#     # .split(tl_model.cfg.d_head, dim=-1)[0],
#     tl_model.W_U,
#     hf_model.lm_head.weight.T,  # Gah!
#     atol=1e-9,
#     rtol=1e-9,  # Weights *are* close.
# )

# # %%

# assert MODE == "ln_f"
# manual_tl_logits = (
#     torch.nn.functional.linear(  # Oh weirdly, works opposite way round to matmul
#         tl_act[0],
#         tl_model.W_U.clone().T,
#         bias=None,
#     )
# )
# manual_hf_logits = torch.nn.functional.linear(
#     hf_act[0],
#     hf_model.lm_head.weight,
#     bias=None,
# )

# # %%

# torch.testing.assert_close(
#     manual_tl_logits,
#     manual_hf_logits,
#     # tl_act[0],
#     # hf_act[0],
#     # tl_model.W_U.T,
#     # hf_model.lm_head.weight,
#     atol=1e-9,
#     rtol=1e-9,  # Weights *are* close.
# )

# # %%

# # Ah
# prefixes = ["hf_", "tl_"]
# import os
# import pickle
# import shutil

# import torch

# wus = [
#     torch.load(f"{prefix}wu.pt").clone().contiguous() for prefix in prefixes
# ]  # Completely magic, this just works
# residuals = [torch.load(f"{prefix}residual.pt").clone() for prefix in prefixes]

# # for prefix, wu in zip(prefixes, wus):
# #     save_file({"wu": wu}, f"{prefix}wu.safetensors")

# # for prefix, residual in zip(prefixes, residuals):
# #     save_file({"residual": residual}, f"{prefix}residual.safetensors")

# # wus = [wu / wu.norm(dim=-1, keepdim=True) for wu in wus]
# # residuals = [residual / residual.norm(dim=-1, keepdim=True) for residual in residuals]

# assert (
#     torch.abs(wus[0] - wus[1]).max().item() < 1e-9
# )  # Lots some by normalizing: 1e-9 used to work!
# assert torch.abs(residuals[0] - residuals[1]).max().item() < 1e-9

# # print(
# #     wus[0].abs().max(), residuals[1].abs().max()
# # )  # There are 1.752 and 225.6747. They go as low as 1e-6 too...

# linears = [
#     torch.nn.functional.linear(  # The linear dimension is 768
#         residual,
#         wu,
#         bias=None,
#     )
#     for residual, wu in zip(residuals, wus)
# ]

# abs_max = torch.abs(linears[0] - linears[1]).max().item()
# assert abs_max < 1e-9, abs_max  # Fails! How

# # Ohhh ...768*224 is >1e5

# # %%
