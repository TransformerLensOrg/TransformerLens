import plotly.graph_objects as go
import numpy as np
from numpy import sin, cos, pi
from typing import List, Tuple, Dict, Union, Optional, Callable, Any
from tqdm import tqdm
import pandas as pd
import torch
import plotly.express as px
import gc
import einops

from ioi_dataset import IOIDataset
from ioi_circuit_extraction import do_circuit_extraction

ALL_COLORS = px.colors.qualitative.Dark2
CLASS_COLORS = {
    "name mover": ALL_COLORS[0],
    "negative": ALL_COLORS[1],
    "s2 inhibition": ALL_COLORS[2],
    "induction": ALL_COLORS[5],
    "duplicate token": ALL_COLORS[3],
    "previous token": ALL_COLORS[6],
    "none": ALL_COLORS[7],
    "distributed name mover": "rgb(27,100,119)",
}


from ioi_circuit_extraction import get_extracted_idx

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


def show_pp(
    m,
    xlabel="",
    ylabel="",
    title="",
    bartitle="",
    animate_axis=None,
    highlight_points=None,
    highlight_name="",
    return_fig=False,
    **kwargs,
):
    """
    Plot a heatmap of the values in the matrix `m`
    """

    if animate_axis is None:
        fig = px.imshow(
            m.T,
            title=title if title else "",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            **kwargs,
        )

    else:
        fig = px.imshow(
            einops.rearrange(m, "a b c -> a c b"),
            title=title if title else "",
            animation_frame=animate_axis,
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            **kwargs,
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

    if highlight_points is not None:
        fig.add_scatter(
            x=highlight_points[1],
            y=highlight_points[0],
            mode="markers",
            marker=dict(color="green", size=10, opacity=0.5),
            name=highlight_name,
        )

    fig.update_layout(
        yaxis_title=ylabel,
        xaxis_title=xlabel,
        xaxis_range=[-0.5, m.shape[0] - 0.5],
        showlegend=True,
        legend=dict(x=-0.1),
    )
    if highlight_points is not None:
        fig.update_yaxes(range=[m.shape[1] - 0.5, -0.5], autorange=False)
    fig.show()
    if return_fig:
        return fig


# Plot attention patterns weighted by value norm


def show_attention_patterns(model, heads, ioi_dataset, mode="val", title_suffix="", return_fig=False, return_mtx=False):
    assert mode in [
        "attn",
        "val",
    ]  # value weighted attention or attn for attention probas
    assert type(ioi_dataset) == IOIDataset

    for (layer, head) in heads:
        cache = {}

        good_names = [f"blocks.{layer}.attn.hook_attn"]
        if mode == "val":
            good_names.append(f"blocks.{layer}.attn.hook_v")
        model.cache_some(cache=cache, names=lambda x: x in good_names)  # shape: batch head_no seq_len seq_len

        logits = model(ioi_dataset.text_prompts)

        for i, text in enumerate(ioi_dataset.text_prompts):
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
            if return_fig and not return_mtx:
                return fig
            elif return_mtx and not return_fig:
                return attn if mode == "attn" else cont
            else:
                fig.show()


def safe_del(a):
    """Try and delete a even if it doesn't yet exist"""
    try:
        exec(f"del {a}")
    except:
        pass
    torch.cuda.empty_cache()


def get_indices_from_sql_file(fname, trial_id):
    """
    Given a SQL file, return the indices of the trial_id
    """
    import sqlite3
    import pandas as pd

    conn = sqlite3.connect(fname)
    df = pd.read_sql_query("SELECT * from trial_params", conn)
    return list(map(int, df[df.trial_id == trial_id].param_value.values))


global last_time
last_time = None
import time


def get_time(s):
    global last_time
    if last_time is None:
        last_time = time.time()
    else:
        print(f"Time elapsed - {s} -: {time.time() - last_time}")
        last_time = time.time()


def scatter_attention_and_contribution(
    model,
    layer_no,
    head_no,
    ioi_dataset,
    return_vals=False,
    return_fig=False,
):
    """
    Plot a scatter plot
    for each input sequence with the attention paid to IO and S
    and the amount that is written in the IO and S directions
    """
    n_heads = model.cfg.n_heads
    n_layers = model.cfg.n_layers
    model_unembed = model.unembed.W_U.detach().cpu()
    df = []
    cache = {}
    model.cache_all(cache)

    logits = model(ioi_dataset.text_prompts)

    for i, prompt in tqdm(enumerate(ioi_dataset.ioi_prompts)):

        io_tok = model.tokenizer(" " + prompt["IO"])["input_ids"][0]
        s_tok = model.tokenizer(" " + prompt["S"])["input_ids"][0]
        toks = model.tokenizer(prompt["text"])["input_ids"]
        io_pos = toks.index(io_tok)
        s1_pos = toks.index(s_tok)
        s2_pos = toks[s1_pos + 1 :].index(s_tok) + (s1_pos + 1)
        assert toks[-1] == io_tok

        io_dir = model_unembed[io_tok].detach()
        s_dir = model_unembed[s_tok].detach()

        # model.reset_hooks() # should allow things to be done with ablated models

        for dire, posses, tok_type in [
            (io_dir, [io_pos], "IO"),
            (s_dir, [s1_pos, s2_pos], "S"),
        ]:
            prob = sum(
                [
                    cache[f"blocks.{layer_no}.attn.hook_attn"][i, head_no, ioi_dataset.word_idx["end"][i], pos]
                    .detach()
                    .cpu()
                    for pos in posses
                ]
            )
            resid = (
                cache[f"blocks.{layer_no}.attn.hook_result"][i, ioi_dataset.word_idx["end"][i], head_no, :]
                .detach()
                .cpu()
            )
            dot = torch.einsum("a,a->", resid, dire)
            df.append([prob, dot, tok_type, prompt["text"]])

    # most of the pandas stuff is intuitive, no need to deeply understand
    viz_df = pd.DataFrame(df, columns=[f"Attn Prob on Name", f"Dot w Name Embed", "Name Type", "text"])
    fig = px.scatter(
        viz_df,
        x=f"Attn Prob on Name",
        y=f"Dot w Name Embed",
        color="Name Type",
        hover_data=["text"],
        color_discrete_sequence=["rgb(114,255,100)", "rgb(201,165,247)"],
        title=f"How Strong {layer_no}.{head_no} Writes in the Name Embed Direction Relative to Attn Prob",
    )

    if return_vals:
        return viz_df
    if return_fig:
        return fig
    else:
        fig.show()


# metrics
# (Callable[ [EasyTransformer, IOIDataset], ...]) # probably a tensor, but with more stuff too as well sometimes


def handle_all_and_std(returning, all, std):
    """
    For use by the below functions. Lots of options!!!
    """

    if all and not std:
        return returning
    if std:
        if all:
            first_bit = (returning).detach().cpu()
        else:
            first_bit = (returning).mean().detach().cpu()
        return first_bit, torch.std(returning).detach().cpu()
    return (returning).mean().detach().cpu()


def logit_diff(model, ioi_dataset, all=False, std=False):
    """
    Difference between the IO and the S logits at the "to" token
    """
    text_prompts = ioi_dataset.text_prompts
    logits = model(text_prompts).detach()
    IO_logits = logits[
        torch.arange(len(text_prompts)),
        ioi_dataset.word_idx["end"],
        ioi_dataset.io_tokenIDs,
    ]
    S_logits = logits[
        torch.arange(len(text_prompts)),
        ioi_dataset.word_idx["end"],
        ioi_dataset.s_tokenIDs,
    ]

    return handle_all_and_std(IO_logits - S_logits, all, std)


def positions(x: torch.Tensor):
    """
    x is a tensor of shape (B, L)
    returns the order of the elements in x
    """
    return torch.argsort(x, dim=1)


def posses(model, ioi_dataset, all=False, std=False):
    """
    Ranking of the IO token in all the tokens
    """
    text_prompts = ioi_dataset.text_prompts
    logits = model(text_prompts).detach().cpu()  # batch * sequence length * vocab_size
    end_logits = logits[torch.arange(len(text_prompts)), ioi_dataset.word_idx["end"], :]  # batch * vocab_size

    positions = torch.argsort(end_logits, dim=1)
    io_positions = positions[torch.arange(len(text_prompts)), ioi_dataset.io_tokenIDs]

    return handle_all_and_std(io_positions, all, std)


def probs(model, ioi_dataset, all=False, std=False, type="io"):
    """
    IO probs
    """

    text_prompts = ioi_dataset.text_prompts
    logits = model(text_prompts).detach().cpu()  # batch * sequence length * vocab_size
    end_logits = logits[torch.arange(len(text_prompts)), ioi_dataset.word_idx["end"], :]  # batch * vocab_size
    end_probs = torch.softmax(end_logits, dim=1)

    if type == "io":
        token_ids = ioi_dataset.io_tokenIDs
    elif type == "s":
        token_ids = ioi_dataset.s_tokenIDs
    else:
        raise ValueError("type must be io or s")
    io_probs = end_probs[torch.arange(ioi_dataset.N), token_ids]

    return handle_all_and_std(io_probs, all, std)


def all_subsets(L: List) -> List[List]:
    """
    Returns all subsets of L
    """
    if len(L) == 0:
        return [[]]
    else:
        rest = all_subsets(L[1:])
        return rest + [[L[0]] + subset for subset in rest]  # thanks copilot


# some ellipse shit


def ellipse_arc(x_center=0, y_center=0, ax1=[1, 0], ax2=[0, 1], a=1, b=1, N=100):
    # x_center, y_center the coordinates of ellipse center
    # ax1 ax2 two orthonormal vectors representing the ellipse axis directions
    # a, b the ellipse parameters
    if abs(np.linalg.norm(ax1) - 1) > 1e-06 or abs(np.linalg.norm(ax2) - 1) > 1e-06:
        raise ValueError("ax1, ax2 must be unit vectors")
    if abs(np.dot(ax1, ax2)) > 1e-06:
        raise ValueError("ax1, ax2 must be orthogonal vectors")
    t = np.linspace(0, 2 * pi, N)
    # ellipse parameterization with respect to a system of axes of directions a1, a2
    xs = a * cos(t)
    ys = b * sin(t)
    # rotation matrix
    R = np.array([ax1, ax2]).T
    # coordinate of the  ellipse points with respect to the system of axes [1, 0], [0,1] with origin (0,0)
    xp, yp = np.dot(R, [xs, ys])
    x = xp + x_center
    y = yp + y_center
    return x, y


def ellipse_wht(mu, sigma):
    """
    Returns x, y and theta of confidence ellipse
    """
    vals, vecs = np.linalg.eigh(sigma)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    theta = np.arctan2(*vecs[:, 0][::-1])  # grr copilot why degrees
    if theta < 0:
        theta += 2 * pi
    width, height = 2 * np.sqrt(vals)
    return width, height, theta


def plot_ellipse(fig, xs, ys, color="MediumPurple", nstd=1, name=""):
    mu = np.mean(xs), np.mean(ys)
    sigma = np.cov(xs, ys)
    w, h, t = ellipse_wht(mu, sigma)
    print(w, h, t)
    w *= nstd
    h *= nstd
    x, y = ellipse_arc(
        x_center=mu[0],
        y_center=mu[1],
        ax1=[cos(t), sin(t)],
        ax2=[-sin(t), cos(t)],
        a=w,
        b=h,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            marker=dict(size=20, color=color),
            name=name,
        )
    )


def get_heads_from_nodes(nodes, ioi_dataset):
    heads_to_keep_tok = {}
    for h, t in nodes:
        if h not in heads_to_keep_tok:
            heads_to_keep_tok[h] = []
        if t not in heads_to_keep_tok[h]:
            heads_to_keep_tok[h].append(t)

    heads_to_keep = {}
    for h in heads_to_keep_tok:
        heads_to_keep[h] = get_extracted_idx(heads_to_keep_tok[h], ioi_dataset)

    return heads_to_keep


def get_heads_from_nodes(nodes, ioi_dataset):
    heads_to_keep_tok = {}
    for h, t in nodes:
        if h not in heads_to_keep_tok:
            heads_to_keep_tok[h] = []
        if t not in heads_to_keep_tok[h]:
            heads_to_keep_tok[h].append(t)

    heads_to_keep = {}
    for h in heads_to_keep_tok:
        heads_to_keep[h] = get_extracted_idx(heads_to_keep_tok[h], ioi_dataset)

    return heads_to_keep


def circuit_from_nodes_logit_diff(model, ioi_dataset, nodes):
    """Take a list of nodes, return the logit diff of the circuit described by the nodes"""
    heads_to_keep = get_heads_from_nodes(nodes, ioi_dataset)
    # print(heads_to_keep)
    model.reset_hooks()
    model, _ = do_circuit_extraction(
        model=model,
        heads_to_keep=heads_to_keep,
        mlps_to_remove={},
        ioi_dataset=ioi_dataset,
    )
    return logit_diff(model, ioi_dataset, all=False)


def basis_change(x, y):
    """
    Change the basis (1, 0) and (0, 1) to the basis
    1/sqrt(2) (1, 1) and 1/sqrt(2) (-1, 1)
    """

    return (x + y) / np.sqrt(2), (y - x) / np.sqrt(2)


def add_arrow(fig, end_point, start_point, color="black"):
    x_start, y_start = start_point
    x_end, y_end = end_point
    fig.add_annotation(
        x=x_start,
        y=y_start,
        ax=x_end,
        ay=y_end,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        text="",  # if you want only the arrow
        showarrow=True,
        arrowhead=3,
        arrowsize=1,
        arrowwidth=1,
        arrowcolor=color,
    )
