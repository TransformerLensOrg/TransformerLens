import torch
import plotly.express as px
import gc
import einops


# utils for circuit extraction


def join_lists(l1, l2):  # l1 is a list of list. l2 a list of int. We add the int from l2 to the lists of l1.
    assert len(l1) == len(l2)
    assert type(l1[0]) == list and type(l2[0]) == int
    l = []
    for i in range(len(l1)):
        l.append(l1[i] + [l2[i]])
    return l


def get_extracted_idx(idx_list: list[str], ioi_dataset):
    int_idx = [[] for i in range(len(ioi_dataset.text_prompts))]
    for idx_name in idx_list:
        int_idx_to_add = [int(x) for x in list(ioi_dataset.word_idx[idx_name])]  # torch to python objects
        int_idx = join_lists(int_idx, int_idx_to_add)
    return int_idx


CIRCUIT = {"name mover": [((9, 6), (9, 9), (10, 0))], 
    "calibration": [((10, 7), (11, 10))], 
    "s2 inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
    "induction": [(5, 5), (5, 8), (5, 9), (6, 9)],
    "duplicate token": [(0, 1), (0, 10), (3, 0)],
    "previous token": [(2, 2), (2, 9), (4, 11)],
}

RELEVANT_TOKENS = {}
for head in CIRCUIT["name mover"] + CIRCUIT["calibration"] + CIRCUIT["s2 inhibition"]:
    RELEVANT_TOKENS[head] = ["end"]

for head in CIRCUIT["induction"]:
    RELEVANT_TOKENS[head] = ["S2"]

for head in CIRCUIT["duplicate token"]:
    RELEVANT_TOKENS[head] = ["S2"]

for head in CIRCUIT["previous token"]:
    RELEVANT_TOKENS[head] = ["S+1", "and"]

def get_heads_circuit(ioi_dataset, excluded_classes=["calibration"], mlp0=False):
    for excluded_class in excluded_classes:
        assert excluded_class in CIRCUIT.keys()

    heads_to_keep = {}

    for circuit_class in CIRCUIT.keys():
        if circuit_class in excluded_classes:
            continue
        for head in CIRCUIT[circuit_class]:
            heads_to_keep[head] = get_extracted_idx(RELEVANT_TOKENS[head], ioi_dataset)

    if mlp0:
        mlps_to_keep = {}
        mlps_to_keep[0] = get_extracted_idx(
            ["IO", "and", "S", "S+1", "S2", "end"], ioi_dataset
        )  # IO, AND, S, S+1, S2, and END
        return heads_to_keep, mlps_to_keep

    return heads_to_keep


def do_circuit_extraction(
    heads_to_remove=None,  # {(2,3) : List[List[int]]: dimensions dataset_size * datapoint_length
    mlps_to_remove=None,  # {2: List[List[int]]: dimensions dataset_size * datapoint_length
    heads_to_keep=None,  # as above for heads
    mlps_to_keep=None,  # as above for mlps
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
        ioi_dataset=ioi_dataset,
        model=model,
    )

    metric = ExperimentMetric(
        metric=logit_diff_target, dataset=ioi_dataset.text_prompts, relative_metric=False
    )  # TODO make dummy metric

    config = AblationConfig(
        abl_type="custom",
        abl_fn=ablation,
        mean_dataset=ioi_dataset.text_prompts,  # TODO nb of prompts useless ?
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



# other utils


def clear_gpu_mem():
    gc.collect()
    torch.cuda.empty_cache()


def show_tokens(tokens, model, return_list=False):
    # Prints the tokens as text, separated by |
    if type(tokens) == str:
        # If we input text, tokenize first
        tokens = model.to_tokens(tokens)
    text_tokens = [model.tokenizer.decode(t) for t in tokens.squeeze()]
    if return_list:
        return text_tokens
    else:
        print("|".join(text_tokens))


def show_pp(m, xlabel="", ylabel="", title="", bartitle="", animate_axis=None):
    """
    Plot a heatmap of the values in the matrix `m`
    """

    if animate_axis is None:
        fig = px.imshow(
            m.T,
            title=title if title else "",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
        )

    else:
        fig = px.imshow(
            einops.rearrange(m, "a b c -> a c b"),
            title=title if title else "",
            animation_frame=animate_axis,
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
    assert mode in [
        "attn",
        "val",
    ]  # value weighted attention or attn for attention probas
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
                xaxis={
                    "side": "top",
                    "ticktext": words,
                    "tickvals": list(range(len(words))),
                    "tickfont": dict(size=8),
                },
                yaxis={
                    "ticktext": words,
                    "tickvals": list(range(len(words))),
                    "tickfont": dict(size=8),
                },
            )

            fig.show()


def safe_del(a):
    """Try and delete a even if it doesn't yet exist"""
    try:
        exec(f"del {a}")
    except:
        pass
    torch.cuda.empty_cache()