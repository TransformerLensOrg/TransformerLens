# %% [markdown]

# # Attribution Patching Demo
# This is an interim research report, giving a whirlwind tour of some unpublished work I did at Anthropic (credit to the then team - Chris Olah, Catherine Olsson, Nelson Elhage and Tristan Hume for help, support, and mentorship!)
# 
# The goal of this work is run activation patching at an industrial scale, by using gradient based attribution to approximate the technique - allow an arbitrary number of patches to be made on two forwards and a single backward pass
# 
# I have had less time than hoped to flesh out this investigation, but am writing up a rough investigation and comparison to standard activation patching on a few tasks to give a sense of the potential of this approach, and where it works vs falls down.
# %% [markdown]
# ## Setup

# Boring setup, no need to read

# %%
from neel.imports import *
import pysvelte
from IPython.display import HTML, Markdown

# %%
model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=True,
)

model.set_use_attn_result(True)

# %% 

Metric = Callable[[TT[T.batch_and_pos_dims, T.d_model]], float]

# %%
prompt_format = [
    "When John and Mary went to the shops,{} gave the bag to",
    "When Tom and James went to the park,{} gave the ball to",
    "When Dan and Sid went to the shops,{} gave an apple to",
    "After Martin and Amy went to the park,{} gave a drink to",
]
names = [
    (" Mary", " John"),
    (" Tom", " James"),
    (" Dan", " Sid"),
    (" Martin", " Amy"),
]
# List of prompts
prompts = []
# List of answers, in the format (correct, incorrect)
answers = []
# List of the token (ie an integer) corresponding to each answer, in the format (correct_token, incorrect_token)
answer_tokens = []
for i in range(len(prompt_format)):
    for j in range(2):
        answers.append((names[i][j], names[i][1 - j]))
        answer_tokens.append(
            (
                model.to_single_token(answers[-1][0]),
                model.to_single_token(answers[-1][1]),
            )
        )
        # Insert the *incorrect* answer to the prompt, making the correct answer the indirect object.
        prompts.append(prompt_format[i].format(answers[-1][1]))
answer_tokens = torch.tensor(answer_tokens).cuda()
print(prompts)
print(answers)
# %%
example_prompt = "After John and Mary went to the store, John gave a bottle of milk to"
example_answer = " Mary"
utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True)

# %%
tokens = model.to_tokens(prompts, prepend_bos=True)
# Move the tokens to the GPU
tokens = tokens.cuda()
batch_size, context_length = tokens.shape
# Run the model and cache all activations
original_logits, cache = model.run_with_cache(tokens)
# %%
def logits_to_ave_logit_diff(logits, answer_tokens, per_prompt=False):
    # Only the final logits are relevant for the answer
    final_logits = logits[:, -1, :]
    answer_logits = final_logits.gather(dim=-1, index=answer_tokens)
    answer_logit_diff = answer_logits[:, 0] - answer_logits[:, 1]
    if per_prompt:
        return answer_logit_diff
    else:
        return answer_logit_diff.mean()

print("Per prompt logit difference:", logits_to_ave_logit_diff(original_logits, answer_tokens, per_prompt=True))
original_average_logit_diff = logits_to_ave_logit_diff(original_logits, answer_tokens)
print("Average logit difference:", logits_to_ave_logit_diff(original_logits, answer_tokens).item())

# %%
corrupted_prompts = []
for i in range(0, len(prompts), 2):
    corrupted_prompts.append(prompts[i+1])
    corrupted_prompts.append(prompts[i])
corrupted_tokens = model.to_tokens(corrupted_prompts, prepend_bos=True)
corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens, return_type="logits")
corrupted_average_logit_diff = logits_to_ave_logit_diff(corrupted_logits, answer_tokens)
print("Corrupted Average Logit Diff", corrupted_average_logit_diff)
print("Clean Average Logit Diff", original_average_logit_diff)
# %% [markdown]
# ### New Content
# %%
SAVED_CLEAN_VALUE = original_average_logit_diff.item()
SAVED_CORRUPTED_VALUE = corrupted_average_logit_diff.item()

def ioi_metric(logits):
    ave_logit_diff = logits_to_ave_logit_diff(logits, answer_tokens)
    normalised_logit_diff = (ave_logit_diff - SAVED_CORRUPTED_VALUE) / (SAVED_CLEAN_VALUE - SAVED_CORRUPTED_VALUE)
    return normalised_logit_diff

print("Fully recovered (clean) is one:", ioi_metric(original_logits))
print("Baseline (corrupted) is zero:", ioi_metric(corrupted_logits))
# %%
def get_cache_fwd_and_bwd(model, tokens, metric):
    model.reset_hooks()
    cache = {}
    def forward_cache_hook(act, hook):
        cache[hook.name] = act.detach()
    model.add_hook(lambda name: True, forward_cache_hook, "fwd")

    grad_cache = {}
    def backward_cache_hook(act, hook):
        grad_cache[hook.name] = act.detach()
    model.add_hook(lambda name: True, backward_cache_hook, "bwd")

    value = metric(model(tokens))
    value.backward()
    model.reset_hooks()
    return value.item(), ActivationCache(cache, model), ActivationCache(grad_cache, model)



clean_tokens = tokens
clean_value, clean_cache, clean_grad_cache = get_cache_fwd_and_bwd(model, clean_tokens, ioi_metric)
print("Clean Value:", clean_value)
print("Clean Activations Cached:", len(clean_cache))
print("Clean Gradients Cached:", len(clean_grad_cache))
corrupted_value, corrupted_cache, corrupted_grad_cache = get_cache_fwd_and_bwd(model, corrupted_tokens, ioi_metric)
print("Corrupted Value:", corrupted_value)
print("Corrupted Activations Cached:", len(corrupted_cache))
print("Corrupted Gradients Cached:", len(corrupted_grad_cache))
# %% [markdown]
# ### Attention Attribution
# The easiest thing to start with is to not even engage with the corrupted tokens/patching, but to look at the attribution of the attention patterns - that is, the linear approximation to what happens if you set each element of the attention pattern to zero. This, as it turns out, is a good proxy to what is going on with each head!

# Note that this is *not* the same as what we will later do with patching. In particular, this does not set up a careful counterfactual! It's a good tool for what's generally going on in this problem, but does not control for eg stuff that systematically boosts John > Mary in general, stuff that says "I should activate the IOI circuit", etc. Though using logit diff as our metric *does*

# Each element of the batch is independent and the metric is an average logit diff, so we can analyse each batch element independently here. We'll look at the first one, and then at the average across the whole batch (note - 4 prompts have indirect object before subject, 4 prompts have it the other way round, making the average pattern harder to interpret - I plot it over the first sequence of tokens as a mildly misleading reference).

# We can compare it to the interpretability in the wild diagram, and basically instantly recover most of the circuit!

# %%
def create_attention_attr(clean_cache, clean_grad_cache) -> TT["batch", "layer", "head_index", "dest", "src"]:
    attention_stack = torch.stack([clean_cache["pattern", l] for l in range(model.cfg.n_layers)], dim=0)
    attention_grad_stack = torch.stack([clean_grad_cache["pattern", l] for l in range(model.cfg.n_layers)], dim=0)
    attention_attr = attention_grad_stack * attention_stack
    attention_attr = einops.rearrange(attention_attr, "layer batch head_index dest src -> batch layer head_index dest src")
    return attention_attr

attention_attr = create_attention_attr(clean_cache, clean_grad_cache)
# %%
HEAD_NAMES = [f"L{l}H{h}" for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)]
HEAD_NAMES_SIGNED = [f"{name}{sign}" for name in HEAD_NAMES for sign in ["+", "-"]]
HEAD_NAMES_QKV = [f"{name}{act_name}" for name in HEAD_NAMES for act_name in ["Q", "K", "V"]]
print(HEAD_NAMES[:5])
print(HEAD_NAMES_SIGNED[:5])
print(HEAD_NAMES_QKV[:5])
# %% [markdown]
# An extremely janky way to plot the attention attribution patterns. We scale them to be in [-1, 1], split each head into a positive and negative part (so all of it is in [0, 1]), and then plot the top 20 head-halves (a head can appear twice!) by the max value of the attribution pattern.

# %%
from circuitsvis.utils.render import render_cdn
def plot_attention_attr(attention_attr, tokens, top_k=20, index=0, title=""):
    if len(tokens.shape)==2:
        tokens = tokens[index]
    if len(attention_attr.shape)==5:
        attention_attr = attention_attr[index]
    attention_attr_pos = attention_attr.clamp(min=-1e-5)
    attention_attr_neg =  - attention_attr.clamp(max=1e-5)
    attention_attr_signed = torch.stack([attention_attr_pos, attention_attr_neg], dim=0)
    attention_attr_signed = einops.rearrange(attention_attr_signed, "sign layer head_index dest src -> (layer head_index sign) dest src")
    attention_attr_signed = attention_attr_signed / attention_attr_signed.max()
    attention_attr_indices = attention_attr_signed.max(-1).values.max(-1).values.argsort(descending=True)
    # print(attention_attr_indices.shape)
    # print(attention_attr_indices)
    attention_attr_signed = attention_attr_signed[attention_attr_indices, :, :]
    head_labels = [HEAD_NAMES_SIGNED[i.item()] for i in attention_attr_indices]

    # display(HTML(render_cdn("AttentionPatterns", tokens=model.to_str_tokens(tokens), attention=attention_attr_signed[:top_k, :, :], head_labels=head_labels[:top_k])))
    if title: display(Markdown("### "+title))
    display(pysvelte.AttentionMulti(tokens=model.to_str_tokens(tokens), attention=attention_attr_signed.permute(1, 2, 0)[:, :, :top_k], head_labels=head_labels[:top_k]))

plot_attention_attr(attention_attr, tokens, index=0, title="Attention Attribution for first sequence")

plot_attention_attr(attention_attr.sum(0), tokens[0], title="Summed Attention Attribution for all sequences")
print("Note: Plotted over first sequence for reference, but pairs have IO and S1 in different positions.")
# %% [markdown]

# ## Attribution Patching

# In the following sections, I will implement various kinds of attribution patching, and then compare them to the activation patching patterns (activation patching code copied from [Exploratory Analysis Demo](https://neelnanda.io/exploratory-analysis-demo))

# ### Residual Stream Patching

# <details><summary>Note: We add up across both d_model and batch (Explanation).</summary>
# We add up along d_model because we're taking the dot product - the derivative *is* the linear map that locally linearly approximates the metric, and so we take the dot product of our change vector with the derivative vector. Equivalent, we look at the effect of changing each coordinate independently, and then combine them by adding it up - it's linear, so this totally works. 
# We add up across batch because we're taking the average of the metric, so each individual batch element provides `1/batch_size` of the overall effect. Because each batch element is independent of the others and no information moves between activations for different inputs, the batched version is equivalent to doing attribution patching separately for each input, and then averaging - in this second version the metric per input is *not* divided by batch_size because we don't average.</details>

# %%
def attr_patch_residual(
        clean_cache: ActivationCache, 
        corrupted_cache: ActivationCache, 
        corrupted_grad_cache: ActivationCache,
    ) -> TT["component", "pos"]:
    clean_residual, residual_labels = clean_cache.accumulated_resid(-1, incl_mid=True, return_labels=True)
    corrupted_residual = corrupted_cache.accumulated_resid(-1, incl_mid=True, return_labels=False)
    corrupted_grad_residual = corrupted_grad_cache.accumulated_resid(-1, incl_mid=True, return_labels=False)
    residual_attr = einops.reduce(
        corrupted_grad_residual * (clean_residual - corrupted_residual),
        "component batch pos d_model -> component pos",
        "sum"
    )
    return residual_attr, residual_labels

residual_attr, residual_labels = attr_patch_residual(clean_cache, corrupted_cache, corrupted_grad_cache)
imshow(residual_attr, y=residual_labels, yaxis="Component", xaxis="Position", title="Residual Attribution Patching")

# ### Layer Output Patching

# %%
def attr_patch_layer_out(
        clean_cache: ActivationCache, 
        corrupted_cache: ActivationCache, 
        corrupted_grad_cache: ActivationCache,
    ) -> TT["component", "pos"]:
    clean_layer_out, labels = clean_cache.decompose_resid(-1, return_labels=True)
    corrupted_layer_out = corrupted_cache.decompose_resid(-1, return_labels=False)
    corrupted_grad_layer_out = corrupted_grad_cache.decompose_resid(-1, return_labels=False)
    layer_out_attr = einops.reduce(
        corrupted_grad_layer_out * (clean_layer_out - corrupted_layer_out),
        "component batch pos d_model -> component pos",
        "sum"
    )
    return layer_out_attr, labels

layer_out_attr, layer_out_labels = attr_patch_layer_out(clean_cache, corrupted_cache, corrupted_grad_cache)
imshow(layer_out_attr, y=layer_out_labels, yaxis="Component", xaxis="Position", title="Layer Output Attribution Patching")

# %%
def attr_patch_head_out(
        clean_cache: ActivationCache, 
        corrupted_cache: ActivationCache, 
        corrupted_grad_cache: ActivationCache,
    ) -> TT["component", "pos"]:
    labels = HEAD_NAMES

    clean_head_out = clean_cache.stack_head_results(-1, return_labels=False)
    corrupted_head_out = corrupted_cache.stack_head_results(-1, return_labels=False)
    corrupted_grad_head_out = corrupted_grad_cache.stack_head_results(-1, return_labels=False)
    head_out_attr = einops.reduce(
        corrupted_grad_head_out * (clean_head_out - corrupted_head_out),
        "component batch pos d_model -> component pos",
        "sum"
    )
    return head_out_attr, labels

head_out_attr, head_out_labels = attr_patch_head_out(clean_cache, corrupted_cache, corrupted_grad_cache)
imshow(head_out_attr, y=head_out_labels, yaxis="Component", xaxis="Position", title="Head Output Attribution Patching")
sum_head_out_attr = einops.reduce(head_out_attr, "(layer head) pos -> layer head", "sum", layer=model.cfg.n_layers, head=model.cfg.n_heads)
imshow(sum_head_out_attr, yaxis="Layer", xaxis="Head Index", title="Head Output Attribution Patching Sum Over Pos")

# %% [markdown]

# ### Head Activation Patching

# Intuitively, a head has three inputs, keys, queries and values. We can patch each of these individually to get a sense for where the important part of each head's input comes from! 

# As a sanity check, we also do this for the mixed value. The result is a linear map of this (`z @ W_O == result`), so this is the same as patching the output of the head.

# We plot both the patch for each head over each position, and summed over position (it tends to be pretty sparse, so the latter is the same)

# %%
from typing_extensions import Literal
def stack_head_vector_from_cache(
        cache, 
        activation_name: Literal["q", "k", "v", "z"]
    ) -> TT["layer_and_head_index", "batch", "pos", "d_head"]:
    """Stacks the head vectors from the cache from a specific activation (key, query, value or mixed_value (z)) into a single tensor."""
    stacked_head_vectors = torch.stack([cache[activation_name, l] for l in range(model.cfg.n_layers)], dim=0)
    stacked_head_vectors = einops.rearrange(
        stacked_head_vectors,
        "layer batch pos head_index d_head -> (layer head_index) batch pos d_head"
    )
    return stacked_head_vectors

def attr_patch_head_vector(
        clean_cache: ActivationCache, 
        corrupted_cache: ActivationCache, 
        corrupted_grad_cache: ActivationCache,
        activation_name: Literal["q", "k", "v", "z"],
    ) -> TT["component", "pos"]:
    labels = HEAD_NAMES

    clean_head_vector = stack_head_vector_from_cache(clean_cache, activation_name)
    corrupted_head_vector = stack_head_vector_from_cache(corrupted_cache, activation_name)
    corrupted_grad_head_vector = stack_head_vector_from_cache(corrupted_grad_cache, activation_name)
    head_vector_attr = einops.reduce(
        corrupted_grad_head_vector * (clean_head_vector - corrupted_head_vector),
        "component batch pos d_head -> component pos",
        "sum"
    )
    return head_vector_attr, labels

head_vector_attr_dict = {}
for activation_name, activation_name_full in [("k", "Key"), ("q", "Query"), ("v", "Value"), ("z", "Mixed Value")]:
    display(Markdown(f"#### {activation_name_full} Head Vector Attribution Patching"))
    head_vector_attr_dict[activation_name], head_vector_labels = attr_patch_head_vector(clean_cache, corrupted_cache, corrupted_grad_cache, activation_name)
    imshow(head_vector_attr_dict[activation_name], y=head_vector_labels, yaxis="Component", xaxis="Position", title=f"{activation_name_full} Attribution Patching")
    sum_head_vector_attr = einops.reduce(head_vector_attr_dict[activation_name], "(layer head) pos -> layer head", "sum", layer=model.cfg.n_layers, head=model.cfg.n_heads)
    imshow(sum_head_vector_attr, yaxis="Layer", xaxis="Head Index", title=f"{activation_name_full} Attribution Patching Sum Over Pos")

# %%
from typing_extensions import Literal
def stack_head_pattern_from_cache(
        cache, 
    ) -> TT["layer_and_head_index", "batch", "dest_pos", "src_pos"]:
    """Stacks the head patterns from the cache into a single tensor."""
    stacked_head_pattern = torch.stack([cache["pattern", l] for l in range(model.cfg.n_layers)], dim=0)
    stacked_head_pattern = einops.rearrange(
        stacked_head_pattern,
        "layer batch head_index dest_pos src_pos -> (layer head_index) batch dest_pos src_pos"
    )
    return stacked_head_pattern

def attr_patch_head_pattern(
        clean_cache: ActivationCache, 
        corrupted_cache: ActivationCache, 
        corrupted_grad_cache: ActivationCache,
    ) -> TT["component", "dest_pos", "src_pos"]:
    labels = HEAD_NAMES

    clean_head_pattern = stack_head_pattern_from_cache(clean_cache)
    corrupted_head_pattern = stack_head_pattern_from_cache(corrupted_cache)
    corrupted_grad_head_pattern = stack_head_pattern_from_cache(corrupted_grad_cache)
    head_pattern_attr = einops.reduce(
        corrupted_grad_head_pattern * (clean_head_pattern - corrupted_head_pattern),
        "component batch dest_pos src_pos -> component dest_pos src_pos",
        "sum"
    )
    return head_pattern_attr, labels

head_pattern_attr, labels = attr_patch_head_pattern(clean_cache, corrupted_cache, corrupted_grad_cache)

plot_attention_attr(einops.rearrange(head_pattern_attr, "(layer head) dest src -> layer head dest src", layer=model.cfg.n_layers, head=model.cfg.n_heads), tokens, index=0, title="Head Pattern Attribution Patching")
# %%
def get_head_vector_grad_input_from_grad_cache(
        grad_cache: ActivationCache, 
        activation_name: Literal["q", "k", "v"],
        layer: int
    ) -> TT["batch", "pos", "head_index", "d_model"]:
    vector_grad = grad_cache[activation_name, layer]
    ln_scales = grad_cache["scale", layer, "ln1"]
    attn_layer_object = model.blocks[layer].attn
    if activation_name == "q":
        W = attn_layer_object.W_Q
    elif activation_name == "k":
        W = attn_layer_object.W_K
    elif activation_name == "v":
        W = attn_layer_object.W_V
    else:
        raise ValueError("Invalid activation name")

    return einsum("batch pos head_index d_head, batch pos, head_index d_model d_head -> batch pos head_index d_model", vector_grad, ln_scales.squeeze(-1), W)

def get_stacked_head_vector_grad_input(grad_cache, activation_name: Literal["q", "k", "v"]) -> TT["layer", "batch", "pos", "head_index", "d_model"]:
    return torch.stack([get_head_vector_grad_input_from_grad_cache(grad_cache, activation_name, l) for l in range(model.cfg.n_layers)], dim=0)

def get_full_vector_grad_input(grad_cache) -> TT["qkv", "layer", "batch", "pos", "head_index", "d_model"]:
    return torch.stack([get_stacked_head_vector_grad_input(grad_cache, activation_name) for activation_name in ['q', 'k', 'v']], dim=0)

def attr_patch_head_path(
        clean_cache: ActivationCache, 
        corrupted_cache: ActivationCache, 
        corrupted_grad_cache: ActivationCache
    ) -> TT["qkv", "dest_component", "src_component", "pos"]:
    """
    Computes the attribution patch along the path between each pair of heads.

    Sets this to zero for the path from any late head to any early head

    """
    start_labels = HEAD_NAMES
    end_labels = HEAD_NAMES_QKV
    full_vector_grad_input = get_full_vector_grad_input(corrupted_grad_cache)
    clean_head_result_stack = clean_cache.stack_head_results(-1)
    corrupted_head_result_stack = corrupted_cache.stack_head_results(-1)
    diff_head_result = einops.rearrange(
        clean_head_result_stack - corrupted_head_result_stack,
        "(layer head_index) batch pos d_model -> layer batch pos head_index d_model",
        layer = model.cfg.n_layers,
        head_index = model.cfg.n_heads,
    )
    path_attr = einsum(
        "qkv layer_end batch pos head_end d_model, layer_start batch pos head_start d_model -> qkv layer_end head_end layer_start head_start pos", 
        full_vector_grad_input, 
        diff_head_result)
    correct_layer_order_mask = (
        torch.arange(model.cfg.n_layers)[None, :, None, None, None, None] > 
        torch.arange(model.cfg.n_layers)[None, None, None, :, None, None]).to(path_attr.device)
    zero = torch.zeros(1, device=path_attr.device)
    path_attr = torch.where(correct_layer_order_mask, path_attr, zero)

    path_attr = einops.rearrange(
        path_attr,
        "qkv layer_end head_end layer_start head_start pos -> (layer_end head_end qkv) (layer_start head_start) pos",
    )
    return path_attr, end_labels, start_labels

head_path_attr, end_labels, start_labels = attr_patch_head_path(clean_cache, corrupted_cache, corrupted_grad_cache)
imshow(head_path_attr.sum(-1), y=end_labels, yaxis="Path End (Head Input)", x=start_labels, xaxis="Path Start (Head Output)", title="Head Path Attribution Patching")


# %% [markdown]

# This is hard to parse. Here's an experiment with filtering for the most important heads and showing their paths.

# %%
head_out_values, head_out_indices  = head_out_attr.sum(-1).abs().sort(descending=True)
line(head_out_values)
top_head_indices = head_out_indices[:22].sort().values
top_end_indices = []
top_end_labels = []
top_start_indices = []
top_start_labels = []
for i in top_head_indices:
    i = i.item()
    top_start_indices.append(i)
    top_start_labels.append(start_labels[i])
    for j in range(3):
        top_end_indices.append(3*i+j)
        top_end_labels.append(end_labels[3*i+j])

imshow(head_path_attr[top_end_indices, :][:, top_start_indices].sum(-1), y=top_end_labels, yaxis="Path End (Head Input)", x=top_start_labels, xaxis="Path Start (Head Output)", title="Head Path Attribution Patching (Filtered for Top Heads)")
# %%
for j, composition_type in enumerate(["Query", "Key", "Value"]):
    imshow(head_path_attr[top_end_indices, :][:, top_start_indices][j::3].sum(-1), y=top_end_labels[j::3], yaxis="Path End (Head Input)", x=top_start_labels, xaxis="Path Start (Head Output)", title=f"Head Path to {composition_type} Attribution Patching (Filtered for Top Heads)")
# %% [markdown]
# ## Validating Attribution vs Activation Patching

# As a low effort sanity check that attribution patching works at all, I've ripped out the activation patching arrays from my [Exploratory Analysis Demo](https://neelnanda.io/exploratory-analysis-demo) notebook, and made scatter plots comparing to the attribution patching outputs. Generally it's a decent approximation! The main place it fails is MLP0 and the residual stream

# My fuzzy intuition is that attribution patching works badly for "big" things which are poorly modelled as linear approximations, and works well for "small" things which are more like incremental changes. Anything involving replacing the embedding is a "big" thing, which includes residual streams, and in GPT-2 small MLP0 seems to be used as an "extended embedding" (where later layers use MLP0's output instead of the token embedding), so I also count it as big.
# %%
d = json.load(open("ioi_patching_data.json", "r"))
patched_residual_stream_diff = torch.tensor(d["patched_residual_stream_diff"])
patched_attn_diff = torch.tensor(d["patched_attn_diff"])
patched_mlp_diff = torch.tensor(d["patched_mlp_diff"])
patched_head_z_diff = torch.tensor(d["patched_head_z_diff"])
patched_head_attn_diff = torch.tensor(d["patched_head_attn_diff"])
# %%
str_tokens = model.to_str_tokens(tokens[0])
scatter(patched_residual_stream_diff.flatten(), residual_attr[:-1:2].flatten(), xaxis="Activation Patched (Exact)", yaxis="Attribution Patched (Approx)", include_diag=True, title="Attr Patch vs Actual Patch (Residual Stream)", color = [l for l in range(model.cfg.n_layers) for p in range(context_length)], hover = [f"Layer {l}, Position {p}, |{str_tokens[p]}|" for l in range(model.cfg.n_layers) for p in range(context_length)])
# %%
scatter(patched_attn_diff.flatten(), layer_out_attr[2::2].flatten(), xaxis="Activation Patched (Exact)", yaxis="Attribution Patched (Approx)", include_diag=True, title="Attr Patch vs Actual Patch (Attention Layer Out)", color = [l for l in range(model.cfg.n_layers) for p in range(context_length)], hover = [f"Layer {l}, Position {p}, |{str_tokens[p]}|" for l in range(model.cfg.n_layers) for p in range(context_length)])
scatter(patched_mlp_diff.flatten(), layer_out_attr[3::2].flatten(), xaxis="Activation Patched (Exact)", yaxis="Attribution Patched (Approx)", include_diag=True, title="Attr Patch vs Actual Patch (MLP Layer Out)", color = [l for l in range(model.cfg.n_layers) for p in range(context_length)], hover = [f"Layer {l}, Position {p}, |{str_tokens[p]}|" for l in range(model.cfg.n_layers) for p in range(context_length)])
# %%
scatter(patched_head_z_diff.flatten(), head_out_attr.sum(dim=-1).flatten(), xaxis="Activation Patched (Exact)", yaxis="Attribution Patched (Approx)", include_diag=True, title="Attr Patch vs Actual Patch (Head Output)", color = [l for l in range(model.cfg.n_layers) for p in range(model.cfg.n_heads)], hover = HEAD_NAMES)
scatter(patched_head_attn_diff.flatten(), head_pattern_attr.sum(dim=[-1, -2]).flatten(), xaxis="Activation Patched (Exact)", yaxis="Attribution Patched (Approx)", include_diag=True, title="Attr Patch vs Actual Patch (Head Pattern (not value))", color = [l for l in range(model.cfg.n_layers) for p in range(model.cfg.n_heads)], hover = HEAD_NAMES)
# %%
# %% [markdown]
# ## Factual Knowledge Patching Example

# %%

gpt2_xl = HookedTransformer.from_pretrained("gpt2-xl")
clean_prompt = "The Eiffel Tower is located in the city of"
clean_answer = " Paris"
# corrupted_prompt = "The red brown fox jumps is located in the city of"
corrupted_prompt = "The Colosseum is located in the city of"
corrupted_answer = " Rome"
utils.test_prompt(clean_prompt, clean_answer, gpt2_xl)
utils.test_prompt(corrupted_prompt, corrupted_answer, gpt2_xl)
# %%
clean_answer_index = gpt2_xl.to_single_token(clean_answer)
corrupted_answer_index = gpt2_xl.to_single_token(corrupted_answer)
def factual_logit_diff(logits: TT["batch", "position", "d_vocab"]):
    return logits[0, -1, clean_answer_index] - logits[0, -1, corrupted_answer_index]

# %%
clean_logits, clean_cache = gpt2_xl.run_with_cache(clean_prompt)
CLEAN_LOGIT_DIFF_FACTUAL = factual_logit_diff(clean_logits).item()
corrupted_logits, _ = gpt2_xl.run_with_cache(corrupted_prompt)
CORRUPTED_LOGIT_DIFF_FACTUAL = factual_logit_diff(corrupted_logits).item()

def factual_metric(logits: TT["batch", "position", "d_vocab"]):
    return (factual_logit_diff(logits) - CORRUPTED_LOGIT_DIFF_FACTUAL) / (CLEAN_LOGIT_DIFF_FACTUAL - CORRUPTED_LOGIT_DIFF_FACTUAL)
print("Clean logit diff:", CLEAN_LOGIT_DIFF_FACTUAL)
print("Corrupted logit diff:", CORRUPTED_LOGIT_DIFF_FACTUAL)
print("Clean Metric:", factual_metric(clean_logits))
print("Corrupted Metric:", factual_metric(corrupted_logits))
# %%
# corrupted_value, corrupted_cache, corrupted_grad_cache = get_cache_fwd_and_bwd(gpt2_xl, corrupted_prompt, factual_metric)
# %%
clean_tokens = gpt2_xl.to_tokens(clean_prompt)
clean_str_tokens = gpt2_xl.to_str_tokens(clean_prompt)
corrupted_tokens = gpt2_xl.to_tokens(corrupted_prompt)
corrupted_str_tokens = gpt2_xl.to_str_tokens(corrupted_prompt)
print("Clean:", clean_str_tokens)
print("Corrupted:", corrupted_str_tokens)
# %%
def act_patch_residual(clean_cache, corrupted_tokens, model: HookedTransformer, metric):
    if len(corrupted_tokens.shape)==2:
        corrupted_tokens = corrupted_tokens[0]
    residual_patches = torch.zeros((model.cfg.n_layers, len(corrupted_tokens)), device=model.cfg.device)
    def residual_hook(resid_pre, hook, layer, pos):
        resid_pre[:, pos, :] = clean_cache["resid_pre", layer][:, pos, :]
        return resid_pre
    for layer in tqdm.tqdm(range(model.cfg.n_layers)):
        for pos in range(len(corrupted_tokens)):
            patched_logits = model.run_with_hooks(corrupted_tokens, fwd_hooks=[(f"blocks.{layer}.hook_resid_pre", partial(residual_hook, layer=layer, pos=pos))])
            residual_patches[layer, pos] = metric(patched_logits).item()
    return residual_patches

residual_act_patch = act_patch_residual(clean_cache, corrupted_tokens, gpt2_xl, factual_metric)

imshow(residual_act_patch, title="Factual Recall Patching (Residual)", xaxis="Position", yaxis="Layer", x=clean_str_tokens)
# %%
